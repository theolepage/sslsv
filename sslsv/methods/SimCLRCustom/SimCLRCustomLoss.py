from typing import Callable, Tuple
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather, get_world_size, get_rank
from sslsv.methods.SimCLR.SimCLRLoss import SimCLRLoss

import math


class BaseLoss(nn.Module):

    def __init__(self, logits_fn: Callable[[T, T], T] = lambda A, B: A @ B.T):
        super().__init__()

        self.logits_fn = logits_fn

    def _determine_logits(
        self,
        A: T,
        B: T,
        discard_identity: bool = True,
    ) -> Tuple[T, T]:
        N, V_A, D = A.size()
        _, V_B, _ = B.size()

        A = A.transpose(0, 1).reshape(V_A * N, D)
        B = B.transpose(0, 1).reshape(V_B * N, D)

        A = F.normalize(A, p=2, dim=-1)
        B = F.normalize(B, p=2, dim=-1)
        logits = self.logits_fn(A, gather(B))
        # logits: (V_A*N, world_size*V_B*N)

        pos_mask, neg_mask = SimCLRLoss.create_contrastive_masks(
            N=N,
            V_A=V_A,
            V_B=V_B,
            rank=get_rank(),
            world_size=get_world_size(),
            discard_identity=discard_identity,
        )

        pos = logits[pos_mask].view(V_A * N, -1)
        neg = logits[neg_mask].view(V_A * N, -1)

        # pos: (V_A*N, V_B) -> V_B positives or V_B-1 positives (if discard_identity)
        # neg: (V_A*N, V_B*N*world_size-V_B) -> V_B*(N*world_size-1) negatives

        return pos, neg

    def forward(self, Z_1: T, Z_2: T) -> T:
        raise NotImplementedError


class MHERegularization(BaseLoss):

    def __init__(self, loss_reg_weight: float):
        super().__init__(logits_fn=lambda A, B: torch.cdist(A, B))

        self.weight = loss_reg_weight

    def forward(self, A: T, B: T, discard_identity: bool = True) -> T:
        _, neg = self._determine_logits(A, B, discard_identity)

        loss = self.weight * torch.mean(1 / (neg**2))

        return loss


class SNTXent(BaseLoss):

    def __init__(self, loss_scale: float):
        super().__init__()

        self.scale = loss_scale

    def _determine_labels(self, logits: T, nb_positives: int) -> T:
        labels = torch.zeros(logits.size(), device=logits.device)
        for i in range(nb_positives):
            labels[:, i] = 1.0
        return labels

    def _process_pos(self, pos: T) -> T:
        return pos

    def _process_neg(self, neg: T) -> T:
        return neg

    def _process_logits(self, logits: T) -> T:
        return self.scale * logits

    def forward(self, A: T, B: T, discard_identity: bool = True) -> T:
        pos, neg = self._determine_logits(A, B, discard_identity)

        logits = torch.cat((self._process_pos(pos), self._process_neg(neg)), dim=1)
        logits = self._process_logits(logits)

        labels = self._determine_labels(logits, nb_positives=pos.size(-1))

        loss = F.cross_entropy(logits, labels)

        return loss


class NTXent(SNTXent):

    def __init__(self, loss_scale: float):
        super().__init__(loss_scale)

    def forward(self, A: T, B: T, discard_identity: bool) -> T:
        return super().forward(A[:, 0:1], B[:, 1:2], discard_identity=False)


class SNTXentAM(SNTXent):

    def __init__(
        self,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_alpha: int,
        loss_margin_simo_K: int,
        loss_margin_learnable: bool,
    ):
        super().__init__(loss_scale)

        self.margin = loss_margin

        if loss_margin_simo:
            tau = 1 / self.scale
            alpha = loss_margin_simo_alpha
            K = loss_margin_simo_K
            self.margin = tau * math.log(alpha / K)

        if loss_margin_learnable:
            self.margin = nn.Parameter(torch.tensor(self.margin))

    def _process_pos(self, pos: T) -> T:
        return pos - self.margin


class SNTXentAAM(SNTXentAM):

    def __init__(
        self,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_alpha: int,
        loss_margin_simo_K: int,
        loss_margin_learnable: bool,
    ):
        super().__init__(
            loss_scale,
            loss_margin,
            loss_margin_simo,
            loss_margin_simo_alpha,
            loss_margin_simo_K,
            loss_margin_learnable,
        )

    def _process_pos(self, pos: T) -> T:
        sine = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))

        phi = pos * math.cos(self.margin) - sine * math.sin(self.margin)

        th = math.cos(math.pi - self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        pos = torch.where((pos - th) > 0, phi, pos - mm)

        return pos


class SimCLRCustomLossEnum(Enum):

    NTXENT = "nt-xent"
    SNTXENT = "snt-xent"
    SNTXENTAM = "snt-xent-am"
    SNTXENTAAM = "snt-xent-aam"


class SimCLRCustomLoss(nn.Module):

    def __init__(
        self,
        enable_multi_views: bool,
        loss: SimCLRCustomLossEnum,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_K: int,
        loss_margin_simo_alpha: int,
        loss_margin_learnable: bool,
        loss_reg_weight: float,
    ):
        super().__init__()

        self.enable_multi_views = enable_multi_views

        self.reg = MHERegularization(loss_reg_weight) if loss_reg_weight > 0 else None

        if loss == SimCLRCustomLossEnum.NTXENT:
            self.loss_fn = NTXent(loss_scale)
        elif loss == SimCLRCustomLossEnum.SNTXENT:
            self.loss_fn = SNTXent(loss_scale)
        elif loss == SimCLRCustomLossEnum.SNTXENTAM:
            self.loss_fn = SNTXentAM(
                loss_scale,
                loss_margin,
                loss_margin_simo,
                loss_margin_simo_K,
                loss_margin_simo_alpha,
                loss_margin_learnable,
            )
        elif loss == SimCLRCustomLossEnum.SNTXENTAAM:
            self.loss_fn = SNTXentAAM(
                loss_scale,
                loss_margin,
                loss_margin_simo,
                loss_margin_simo_K,
                loss_margin_simo_alpha,
                loss_margin_learnable,
            )
        else:
            raise NotImplementedError

    def forward(self, Z: T) -> T:
        # Z shape: (N, V, C)

        GLOBAL_VIEWS = Z[:, :2]
        LOCAL_VIEWS = Z[:, 2:]

        loss = self.loss_fn(GLOBAL_VIEWS, GLOBAL_VIEWS, discard_identity=True)
        if self.reg:
            loss += self.reg(GLOBAL_VIEWS, GLOBAL_VIEWS, discard_identity=True)

        if self.enable_multi_views:
            loss += self.loss_fn(LOCAL_VIEWS, GLOBAL_VIEWS, discard_identity=False)
            if self.reg:
                loss += self.reg(LOCAL_VIEWS, GLOBAL_VIEWS, discard_identity=False)

        return loss
