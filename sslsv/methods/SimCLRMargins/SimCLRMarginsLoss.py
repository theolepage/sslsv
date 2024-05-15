from typing import Callable, Tuple
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather, get_world_size, get_rank
from sslsv.methods.SimCLR.SimCLRLoss import SimCLRLoss

import math


class BaseContrastiveLoss(nn.Module):
    """
    Base loss for contrastive-based objective functions.

    Attributes:
        logits_fn (Callable[[T, T], T]): Function that determines similarity of logits.
            Defaults to A @ B.T (dot product).
    """

    def __init__(self, logits_fn: Callable[[T, T], T] = lambda A, B: A @ B.T):
        """
        Initialize a Base Contrastive loss.

        Args:
            logits_fn (Callable[[T, T], T]): Function that determines similarity of logits.

        Returns:
            None
        """
        super().__init__()

        self.logits_fn = logits_fn

    def _determine_logits(
        self,
        A: T,
        B: T,
        discard_identity: bool = True,
    ) -> Tuple[T, T]:
        """
        Determine the positives and negatives for a contrastive loss.

        Args:
            A (T): First embeddings tensor. Shape: (N, V_A, D).
            B (T): Second embeddings tensor. Shape: (N, V_B, D).
            discard_identity (bool): Whether to discard identity comparisons. Defaults to True.

        Returns:
            Tuple[T, T]: Positive and negative logits tensors.
        """
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
        """
        Compute loss.

        Args:
            Z_1 (T): First embeddings tensor.
            Z_2 (T): Second embeddings tensor.

        Returns:
            T: Loss tensor.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError


class MHERegularization(BaseContrastiveLoss):
    """
    MHE (Minimum Hyperspherical Energy) regularization loss.

    Attributes:
        weight (float): Weight for the regularization term.
    """

    def __init__(self, loss_reg_weight: float):
        """
        Initialize an MHERegularization loss.

        Args:
            loss_reg_weight (float): Weight for regularization loss.

        Returns:
            None
        """
        super().__init__(logits_fn=lambda A, B: torch.cdist(A, B))

        self.weight = loss_reg_weight

    def forward(self, A: T, B: T, discard_identity: bool = True) -> T:
        """
        Compute loss.

        Args:
            A (T): First embeddings tensor.
            B (T): First embeddings tensor.
            discard_identity (bool): Whether to discard identity comparisons. Defaults to True.

        Returns:
            T: Loss tensor.
        """
        _, neg = self._determine_logits(A, B, discard_identity)

        loss = self.weight * torch.mean(1 / (neg**2))

        return loss


class SNTXent(BaseContrastiveLoss):
    """
    Symmetric NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Attributes:
        scale (float): Scale factor (1 / temperature).
    """

    def __init__(self, loss_scale: float):
        """
        Initialize a Symmetric NT-Xent loss.

        Args:
            loss_scale (float): Scale factor (1 / temperature).

        Returns:
            None
        """
        super().__init__()

        self.scale = loss_scale

    def _determine_labels(self, logits: T, nb_positives: int) -> T:
        """
        Determines the labels.

        Args:
            logits (T): Logits tensor.
            nb_positives (int): Number of positive samples.

        Returns:
            T: Labels tensor.
        """
        labels = torch.zeros(logits.size(), device=logits.device)
        for i in range(nb_positives):
            labels[:, i] = 1.0
        return labels

    def _process_pos(self, pos: T) -> T:
        """
        Processes positives.

        Args:
            pos (T): Positives tensor.

        Returns:
            T: Positives tensor.
        """
        return pos

    def _process_neg(self, neg: T) -> T:
        """
        Process negatives.

        Args:
            neg (T): Negatives tensor.

        Returns:
            T: Negatives tensor.
        """
        return neg

    def _process_logits(self, logits: T) -> T:
        """
        Process logits.

        Args:
            logits (T): Logits tensor.

        Returns:
            T: Logits tensor.
        """
        return self.scale * logits

    def forward(self, A: T, B: T, discard_identity: bool = True) -> T:
        """
        Compute loss.

        Args:
            A (T): First embeddings tensor.
            B (T): Second embeddings tensor.
            discard_identity (bool): Whether to discard identity comparisons. Defaults to True.

        Returns:
            T: Loss tensor.
        """
        pos, neg = self._determine_logits(A, B, discard_identity)

        logits = torch.cat((self._process_pos(pos), self._process_neg(neg)), dim=1)
        logits = self._process_logits(logits)

        labels = self._determine_labels(logits, nb_positives=pos.size(-1))

        loss = F.cross_entropy(logits, labels)

        return loss


class NTXent(SNTXent):
    """
    NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss.
    """

    def __init__(self, loss_scale: float):
        """
        Initialize an NT-Xent loss.

        Args:
            loss_scale (float): Scale factor (1 / temperature).

        Returns:
            None
        """
        super().__init__(loss_scale)

    def forward(self, A: T, B: T, discard_identity: bool) -> T:
        """
        Compute loss.

        Args:
            A (T): First embeddings tensor.
            B (T): Second embeddings tensor.
            discard_identity (bool): Whether to discard identity comparisons. Defaults to True.

        Returns:
            T: Loss tensor.
        """
        return super().forward(A[:, 0:1], B[:, 1:2], discard_identity=False)


class SNTXentAM(SNTXent):
    """
    SNT-Xent-AM (Additive Margins) loss.

    Attributes:
        margin (Union[float, nn.Parameter]): Margin value or parameter.
    """

    def __init__(
        self,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_K: int,
        loss_margin_simo_alpha: int,
        loss_margin_learnable: bool,
    ):
        """
        Initialize an SNT-Xent-AM loss.

        Args:
            loss_scale (float): Scale factor (1 / temperature).
            loss_margin (float): Margin value.
            loss_margin_simo (bool): Whether to use SIMO as the margin.
            loss_margin_simo_K (int): K value for the SIMO margin.
            loss_margin_simo_alpha (int): Alpha value for the SIMO margin.
            loss_margin_learnable (bool): Whether the margin value is learnable.

        Returns:
            None
        """
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
        """
        Process positives (apply Additive Margins).

        Args:
            pos (T): Positives tensor.

        Returns:
            T: Positives tensor.
        """
        return pos - self.margin


class SNTXentAAM(SNTXentAM):
    """
    SNT-Xent-AAM (Additive Angular Margins) loss.
    """

    def __init__(
        self,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_K: int,
        loss_margin_simo_alpha: int,
        loss_margin_learnable: bool,
    ):
        """
        Initialize an SNT-Xent-AAM loss.

        Args:
            loss_scale (float): Scale factor (1 / temperature).
            loss_margin (float): Margin value.
            loss_margin_simo (bool): Whether to use SIMO as the margin.
            loss_margin_simo_K (int): K value for the SIMO margin.
            loss_margin_simo_alpha (int): Alpha value for the SIMO margin.
            loss_margin_learnable (bool): Whether the margin value is learnable.

        Returns:
            None
        """
        super().__init__(
            loss_scale,
            loss_margin,
            loss_margin_simo,
            loss_margin_simo_alpha,
            loss_margin_simo_K,
            loss_margin_learnable,
        )

    def _process_pos(self, pos: T) -> T:
        """
        Process positives (apply Additive Angular Margins).

        Args:
            pos (T): Positives tensor.

        Returns:
            T: Positives tensor.
        """
        sine = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))

        phi = pos * math.cos(self.margin) - sine * math.sin(self.margin)

        th = math.cos(math.pi - self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        pos = torch.where((pos - th) > 0, phi, pos - mm)

        return pos


class SimCLRMarginsLossEnum(Enum):
    """
    Enumeration representing loss options for SimCLR Margins method.

    Options:
        NTXENT (str): NT-Xent loss.
        SNTXENT (str): SNT-Xent loss.
        SNTXENTAM (str): SNT-Xent loss with angular margin.
        SNTXENTAAM (str): SNT-Xent loss with additive angular margin.
    """

    NTXENT = "nt-xent"
    SNTXENT = "snt-xent"
    SNTXENTAM = "snt-xent-am"
    SNTXENTAAM = "snt-xent-aam"


class SimCLRMarginsLoss(nn.Module):
    """
    SimCLR Margins loss.

    Attributes:
        enable_multi_views (bool): Whether to enable multiple views training.
        reg (MHERegularization): Regularization loss function.
        loss_fn (LossFunction): Loss function.
    """

    def __init__(
        self,
        enable_multi_views: bool,
        loss: SimCLRMarginsLossEnum,
        loss_scale: float,
        loss_margin: float,
        loss_margin_simo: bool,
        loss_margin_simo_K: int,
        loss_margin_simo_alpha: int,
        loss_margin_learnable: bool,
        loss_reg_weight: float,
    ):
        """
        Initialize a SimCLR Margins loss.

        Args:
            enable_multi_views (bool): Whether to enable multiple views training.
            loss (SimCLRMarginsLossEnum): Type of loss function.
            loss_scale (float): Scale factor.
            loss_margin (float): Margin value.
            loss_margin_simo (bool): Whether to use SIMO as the margin.
            loss_margin_simo_K (int): K value for the SIMO margin.
            loss_margin_simo_alpha (int): Alpha value for the SIMO margin.
            loss_margin_learnable (bool): Whether the margin value is learnable.
            loss_reg_weight (float): Weight of the MHE regularization term.

        Returns:
            None

        Raises:
            NotImplementedError: If an unsupported loss function is specified.
        """
        super().__init__()

        self.enable_multi_views = enable_multi_views

        self.reg = MHERegularization(loss_reg_weight) if loss_reg_weight > 0 else None

        if loss == SimCLRMarginsLossEnum.NTXENT:
            self.loss_fn = NTXent(loss_scale)
        elif loss == SimCLRMarginsLossEnum.SNTXENT:
            self.loss_fn = SNTXent(loss_scale)
        elif loss == SimCLRMarginsLossEnum.SNTXENTAM:
            self.loss_fn = SNTXentAM(
                loss_scale,
                loss_margin,
                loss_margin_simo,
                loss_margin_simo_K,
                loss_margin_simo_alpha,
                loss_margin_learnable,
            )
        elif loss == SimCLRMarginsLossEnum.SNTXENTAAM:
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
        """
        Compute loss.

        Args:
            Z (T): Embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        # Z shape: (N, V, C)

        GLOBAL_VIEWS = Z[:, :2]
        LOCAL_VIEWS = Z[:, 2:]

        if self.enable_multi_views:
            loss = self.loss_fn(LOCAL_VIEWS, GLOBAL_VIEWS, discard_identity=False)
            if self.reg:
                loss += self.reg(LOCAL_VIEWS, GLOBAL_VIEWS, discard_identity=False)
        else:
            loss = self.loss_fn(GLOBAL_VIEWS, GLOBAL_VIEWS, discard_identity=True)
            if self.reg:
                loss += self.reg(GLOBAL_VIEWS, GLOBAL_VIEWS, discard_identity=True)

        return loss
