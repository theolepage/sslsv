import torch
from torch import nn
import torch.nn.functional as F

from enum import Enum

from sslsv.utils.distributed import gather, get_world_size, get_rank
from sslsv.methods.SimCLR.SimCLRLoss import SimCLRLoss

import math


class BaseLoss(nn.Module):

    def __init__(self, logits_fn=lambda A, B : A @ B.T):
        super().__init__()

        self.logits_fn = logits_fn

    def _determine_logits(self, A, B, discard_identity=True):
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
            discard_identity=discard_identity
        )

        pos = logits[pos_mask].view(V_A * N, -1)
        neg = logits[neg_mask].view(V_A * N, -1)

        # pos: (V_A*N, V_B) -> V_B positives or V_B-1 positives (if discard_identity)
        # neg: (V_A*N, V_B*N*world_size-V_B) -> V_B*(N*world_size-1) negatives 

        return pos, neg

    def forward(self, Z_1, Z_2):
        raise NotImplementedError


class MHERegularization(BaseLoss):

    def __init__(self, config):
        super().__init__(logits_fn=lambda A, B : torch.cdist(A, B))

        self.weight = config.loss_reg_weight

    def forward(self, A, B, discard_identity=True):
        _, neg = self._determine_logits(A, B, discard_identity)

        loss = self.weight * torch.mean(1 / (neg ** 2))

        return loss


class SNTXent(BaseLoss):

    def __init__(self, config):
        super().__init__()

        self.scale = config.loss_scale

    def _determine_labels(self, logits, nb_positives):
        labels = torch.zeros(logits.size(), device=logits.device)
        for i in range(nb_positives): labels[:, i] = 1.0
        return labels

    def _process_pos(self, pos):
        return pos

    def _process_neg(self, neg):
        return neg

    def _process_logits(self, logits):
        return self.scale * logits

    def forward(self, A, B, discard_identity=True):
        pos, neg = self._determine_logits(A, B, discard_identity)

        logits = torch.cat((
            self._process_pos(pos),
            self._process_neg(neg)
        ), dim=1)
        logits = self._process_logits(logits)

        labels = self._determine_labels(
            logits,
            nb_positives=pos.size(-1)
        )

        loss = F.cross_entropy(logits, labels)

        return loss


class NTXent(SNTXent):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, A, B, discard_identity):
       return super().forward(A[:, 0:1], B[:, 1:2], discard_identity=False)


class SNTXentAM(SNTXent):

    def __init__(self, config):
        super().__init__(config)

        self.margin = config.loss_margin
        self.scale = config.loss_scale

        if config.loss_margin_simo:
            tau = 1 / self.scale
            alpha = config.loss_margin_simo_alpha
            K = config.loss_margin_simo_K
            self.margin = tau * math.log(alpha / K)

        if config.loss_margin_learnable:
            self.margin = nn.Parameter(torch.tensor(self.margin))

    def _process_pos(self, pos):
        return pos - self.margin


class SNTXentAAM(SNTXentAM):

    def __init__(self, config):
        super().__init__(config)

    def _process_pos(self, pos):
        sine = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))

        phi = pos * math.cos(self.margin) - sine * math.sin(self.margin)
        
        th = math.cos(math.pi - self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        pos = torch.where((pos - th) > 0, phi, pos - mm)

        return pos


class SimCLRCustomLossEnum(Enum):

    NTXENT     = 'nt-xent'
    SNTXENT    = 'snt-xent'
    SNTXENTAM  = 'snt-xent-am'
    SNTXENTAAM = 'snt-xent-aam'


class SimCLRCustomLoss(nn.Module):

    _LOSS_METHODS = {
        SimCLRCustomLossEnum.NTXENT     : NTXent,
        SimCLRCustomLossEnum.SNTXENT    : SNTXent,
        SimCLRCustomLossEnum.SNTXENTAM  : SNTXentAM,
        SimCLRCustomLossEnum.SNTXENTAAM : SNTXentAAM,
    }

    def __init__(self, config):
        super().__init__()

        self.enable_multi_views = config.enable_multi_views

        self.reg = (
            MHERegularization(config)
            if config.loss_reg_weight > 0
            else None
        )

        self.loss_fn = self._LOSS_METHODS[config.loss](config)

    def forward(self, Z):
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