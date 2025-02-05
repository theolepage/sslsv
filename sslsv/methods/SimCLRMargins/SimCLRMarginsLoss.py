from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather, get_world_size, get_rank
from sslsv.methods.SimCLR.SimCLRLoss import SimCLRLoss

import math


class SimCLRMarginsLossEnum(Enum):
    """
    Enumeration representing loss options for SimCLR Margins method.

    Options:
        NTXENT (str): NT-Xent loss.
        NTXENT_SPHEREFACE (str): NT-Xent loss with SphereFace.
        NTXENT_COSFACE (str): NT-Xent loss with CosFace (Additive Margin).
        NTXENT_ARCFACE (str): NT-Xent loss with ArcFace (Additive Angular Margin).
        NTXENT_CURRICULARFACE (str): NT-Xent loss with CurricularFace.
        NTXENT_MAGFACE (str): NT-Xent loss with MagFace.
        NTXENT_ADAFACE (str): NT-Xent loss with AdaFace.
    """

    NTXENT = "nt-xent"
    NTXENT_SPHEREFACE = "nt-xent-sphereface"
    NTXENT_COSFACE = "nt-xent-cosface"
    NTXENT_ARCFACE = "nt-xent-arcface"
    NTXENT_CURRICULARFACE = "nt-xent-curricularface"
    NTXENT_MAGFACE = "nt-xent-magface"
    NTXENT_ADAFACE = "nt-xent-adaface"


@dataclass
class SimCLRMarginsLossConfig:
    """
    SimCLR Margins loss configuration.

    Attributes:
        loss (SimCLRMarginsLossEnum): Type of loss function.
        symmetric (bool): Whether to use symmetric formulation of NT-Xent.
        scale (float): Scale factor for the loss function.
        margin (float): Margin value for the loss function.
        margin_learnable (bool): Whether the margin value is learnable.
        margin_scheduler (bool): Whether to use a scheduler for the margin value.
        margin_simo (bool): Whether to use SIMO as the margin.
        margin_simo_K (int): K value for the SIMO margin.
        margin_simo_alpha (int): Alpha value for the SIMO margin.
        magface_l_margin (float): MagFace lower margin.
        magface_u_margin (float): MagFace lpper margin.
        magface_l_a (int): MagFace lower norm.
        magface_u_a (int): MagFace upper norm.
        magface_lambda_g (float): MagFace weight for regularization.
        adaface_h (float): AdaFace hyper-parameter h.
    """

    loss: SimCLRMarginsLossEnum = SimCLRMarginsLossEnum.NTXENT

    symmetric: bool = True

    scale: float = 33.3333333333

    margin: float = 0.1
    margin_learnable: bool = False
    margin_scheduler: bool = False

    # SIMO
    margin_simo: bool = False
    margin_simo_K: int = 2 * 255
    margin_simo_alpha: int = 65536

    # MagFace
    magface_l_margin: float = 0.01
    magface_u_margin: float = 0.05
    magface_l_a: int = 10
    magface_u_a: int = 110
    magface_lambda_g: float = 0

    # AdaFace
    adaface_h: float = 0.333


class SimCLRNTXent(nn.Module):
    """
    NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss.

    Attributes:
        config (SimCLRMarginsLossConfig): Loss configuration.
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__()

        self.config = config

        self.additional_loss = 0

    def _determine_logits(
        self,
        A: T,
        B: T,
        discard_identity: bool = True,
        ssps_assignments: Optional[T] = None,
        logits_fn: Callable[[T, T], T] = lambda A, B: A @ B.T,
    ) -> Tuple[T, T]:
        """
        Determine the positives and negatives for a contrastive loss.

        Args:
            A (T): First embeddings tensor. Shape: (N, V_A, D).
            B (T): Second embeddings tensor. Shape: (N, V_B, D).
            discard_identity (bool): Whether to discard identity comparisons. Defaults to True.
            ssps_assignments (Optional[T]): SSPS assignments to prevent false negatives.
            logits_fn (Callable[[T, T], T]): Function that determines similarity of logits.
                Defaults to A @ B.T (dot product).

        Returns:
            Tuple[T, T]: Positive and negative logits tensors.
        """
        N, V_A, D = A.size()
        _, V_B, _ = B.size()

        A = A.transpose(0, 1).reshape(V_A * N, D)
        B = B.transpose(0, 1).reshape(V_B * N, D)

        A = F.normalize(A, p=2, dim=-1)
        B = F.normalize(B, p=2, dim=-1)
        logits = logits_fn(A, gather(B))
        # logits: (V_A*N, world_size*V_B*N)

        pos_mask, neg_mask = SimCLRLoss.create_contrastive_masks(
            N=N,
            V_A=V_A,
            V_B=V_B,
            rank=get_rank(),
            world_size=get_world_size(),
            discard_identity=discard_identity,
        )

        if ssps_assignments is not None:
            assignments = ssps_assignments.repeat(2)
            assignments_all = gather(assignments)
            assignments_mask = assignments_all.unsqueeze(0) == assignments.unsqueeze(1)
            assignments_mask[~neg_mask] = False
            logits[assignments_mask] = 0

        pos = logits[pos_mask].view(V_A * N, -1)
        neg = logits[neg_mask].view(V_A * N, -1)

        # pos: (V_A*N, V_B) -> V_B positives or V_B-1 positives (if discard_identity)
        # neg: (V_A*N, V_B*N*world_size-V_B) -> V_B*(N*world_size-1) negatives

        return pos, neg

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

    def _process_pos(self, pos: T, norms: T = None) -> T:
        """
        Processes positives.

        Args:
            pos (T): Positives tensor.
            norms (Optional[T]): Norms tensor.

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
        return self.config.scale * logits

    def _compute_loss(
        self,
        A: T,
        B: T,
        discard_identity: bool,
        ssps_assignments: Optional[T] = None,
    ):
        """
        Determine and process logits to compute loss.

        Args:
            A (T): First embeddings tensor.
            B (T): Second embeddings tensor.
            discard_identity (bool): Whether to discard identity comparisons.
            ssps_assignments (Optional[T]): SSPS assignments to prevent false negatives.

        Returns:
            T: Loss tensor.
        """
        pos, neg = self._determine_logits(A, B, discard_identity, ssps_assignments)

        # Compute norms of A
        N, V_A, D = A.size()
        A = A.transpose(0, 1).reshape(V_A * N, D)
        A_norms = torch.norm(A, p=2, dim=1, keepdim=True)  # (V_A * N, 1)
        A_norms = A_norms.repeat(1, pos.size(-1))

        logits = torch.cat(
            (
                self._process_pos(pos, norms=A_norms),
                self._process_neg(neg),
            ),
            dim=1,
        )
        logits = self._process_logits(logits)

        labels = self._determine_labels(logits, nb_positives=pos.size(-1))

        loss = F.cross_entropy(logits, labels)
        loss += self.additional_loss

        return loss

    def forward(
        self,
        A: T,
        B: T,
        discard_identity: bool,
        ssps_assignments: Optional[T] = None,
    ) -> T:
        """
        Compute loss.

        Args:
            A (T): First embeddings tensor.
            B (T): Second embeddings tensor.
            discard_identity (bool): Whether to discard identity comparisons.
            ssps_assignments (Optional[T]): SSPS assignments to prevent false negatives.

        Returns:
            T: Loss tensor.
        """
        if self.config.symmetric:
            return self._compute_loss(
                A,
                B,
                discard_identity,
                ssps_assignments=ssps_assignments,
            )

        return self._compute_loss(
            A[:, 0:1],
            B[:, 1:2],
            discard_identity=False,
            ssps_assignments=ssps_assignments,
        )


class SimCLRNTXentSphereFace(SimCLRNTXent):
    """
    NT-Xent-SphereFace loss.

    Paper:
        SphereFace: Deep Hypersphere Embedding for Face Recognition
        *Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song*
        CVPR 2017
        https://arxiv.org/abs/1704.08063

    Attributes:
        margin (Union[float, nn.Parameter]): Margin value or parameter.
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-SphereFace loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.margin = self.config.margin

    def _process_logits(self, logits: T) -> T:
        """
        Process logits.

        Args:
            logits (T): Logits tensor.

        Returns:
            T: Logits tensor.
        """
        with torch.no_grad():
            m_theta = torch.acos(logits.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
            m_theta[:, 0] = m_theta[:, 0] * self.margin
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2.0 * k
            d_theta = phi_theta - logits

        return self.config.scale * (logits + d_theta)


class SimCLRNTXentCosFace(SimCLRNTXent):
    """
    NT-Xent-CosFace (Additive Margin) loss.

    Paper:
        CosFace: Large Margin Cosine Loss for Deep Face Recognition
        *Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu*
        CVPR 2018
        https://arxiv.org/abs/1801.09414

    Attributes:
        margin (Union[float, nn.Parameter]): Margin value or parameter.
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-CosFace loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.margin = self.config.margin

        if self.config.margin_simo:
            tau = 1 / self.config.scale
            alpha = self.config.margin_simo_alpha
            K = self.config.margin_simo_K
            self.margin = tau * math.log(alpha / K)

        if self.config.margin_learnable:
            self.margin = nn.Parameter(torch.tensor(self.margin))

    def _process_pos(self, pos: T, norms: T = None) -> T:
        """
        Process positives (apply Additive Margin).

        Args:
            pos (T): Positives tensor.
            norms (Optional[T]): Norms tensor.

        Returns:
            T: Positives tensor.
        """
        return pos - self.margin


class SimCLRNTXentArcFace(SimCLRNTXentCosFace):
    """
    NT-Xent-ArcFace (Additive Angular Margin) loss.

    Paper:
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        *Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou*
        CVPR 2019
        https://arxiv.org/abs/1801.09414

    Attributes:
        margin (Union[float, nn.Parameter]): Margin value or parameter.
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-ArcFace loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

    @staticmethod
    def _add_aam(x: T, margin: Union[float, T]) -> T:
        """
        Apply Additive Angular Margin.

        Args:
            x (T): Input tensor.
            margin (float): Margin value or margin tensor.

        Returns:
            T: Output tensor.
        """
        if isinstance(margin, float):
            margin = torch.tensor([margin]).to(x.device)

        sine = torch.sqrt((1.0 - torch.mul(x, x)).clamp(0, 1))

        phi = x * torch.cos(margin) - sine * torch.sin(margin)

        th = torch.cos(math.pi - margin)
        mm = torch.sin(math.pi - margin) * margin
        x = torch.where((x - th) > 0, phi, x - mm)

        return x

    def _process_pos(self, pos: T, norms: T = None) -> T:
        """
        Process positives (apply Additive Angular Margin).

        Args:
            pos (T): Positives tensor.
            norms (Optional[T]): Norms tensor.

        Returns:
            T: Positives tensor.
        """
        return SimCLRNTXentArcFace._add_aam(pos, self.margin)


class SimCLRNTXentCurricularFace(SimCLRNTXent):
    """
    NT-Xent-CurricularFace loss.

    Paper:
        CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
        *Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang*
        CVPR 2020
        https://arxiv.org/abs/2004.00288
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-CurricularFace loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.register_buffer("t", torch.zeros(1))

    def _process_logits(self, logits: T) -> T:
        """
        Process logits.

        Args:
            logits (T): Logits tensor.

        Returns:
            T: Logits tensor.
        """
        m = self.config.margin
        pos = logits[:, 0].unsqueeze(-1).clone()
        neg = logits[:, 1:]

        # Update t
        with torch.no_grad():
            self.t = pos.mean() * 0.01 + (1 - 0.01) * self.t

        # T (positives)
        sin_theta = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))
        cos_theta_m = pos * math.cos(m) - sin_theta * math.sin(m)
        threshold = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m
        pos = torch.where(pos > threshold, cos_theta_m, pos - mm)

        # N (negatives)
        mask = neg > cos_theta_m
        neg[mask] = neg[mask] * (self.t + neg[mask])

        logits = torch.cat((pos, neg), dim=1)

        return self.config.scale * logits


class SimCLRNTXentMagFace(SimCLRNTXent):
    """
    NT-Xent-MagFace loss.

    Paper:
        MagFace: A Universal Representation for Face Recognition and Quality Assessment
        *Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou*
        CVPR 2021
        https://arxiv.org/abs/2103.06627
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-MagFace loss.

        Args:
            config (SimCLR): SimCLRMarginsLossConfig.

        Returns:
            None
        """
        super().__init__(config)

        self.l_margin = config.magface_l_margin
        self.u_margin = config.magface_u_margin
        self.l_a = config.magface_l_a
        self.u_a = config.magface_u_a
        self.lambda_g = config.magface_lambda_g

    def _m(self, x_norm: T) -> T:
        """
        Compute margin values.

        Args:
            x_norm (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        return (self.u_margin - self.l_margin) / (
            self.u_a - self.l_a
        ) * (x_norm - self.l_a) + self.l_margin

    def _g(self, x_norm: T) -> T:
        """
        Compute regularization loss term.

        Args:
            x_norm (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        g = 1 / (self.u_a**2) * x_norm + 1 / x_norm
        return torch.mean(g)

    def _process_pos(self, pos: T, norms: T = None) -> T:
        """
        Process positives.

        Args:
            pos (T): Positives tensor.
            norms (Optional[T]): Norms tensor.

        Returns:
            T: Positives tensor.
        """
        norms = torch.clip(norms, min=self.l_a, max=self.u_a)

        ada_margins = self._m(norms)

        self.additional_loss = self.lambda_g * self._g(norms)

        return SimCLRNTXentArcFace._add_aam(pos, ada_margins)


class SimCLRNTXentAdaFace(SimCLRNTXent):
    """
    NT-Xent-AdaFace loss.

    Paper:
        AdaFace: Quality Adaptive Margin for Face Recognition
        *Minchul Kim, Anil K. Jain, Xiaoming Liu*
        CVPR 2022
        https://arxiv.org/abs/2204.00964
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize an NT-Xent-AdaFace loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.h = config.adaface_h

        self.t_alpha = 0.01
        self.eps = 1e-3

        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def _process_pos(self, pos: T, norms: T = None) -> T:
        """
        Process positives.

        Args:
            pos (T): Positives tensor.
            norms (Optional[T]): Norms tensor.

        Returns:
            T: Positives tensor.
        """
        norms = torch.clip(norms, min=0.001, max=100).clone().detach()

        with torch.no_grad():
            self.batch_mean = (
                norms.mean() * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            )
            self.batch_std = (
                norms.std() * self.t_alpha + (1 - self.t_alpha) * self.batch_std
            )

        margin_scaler = (norms - self.batch_mean) / (self.batch_std + self.eps) * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        g_angular = self.config.margin * margin_scaler * -1
        theta = pos.acos()
        theta_m = torch.clip(theta + g_angular, min=self.eps, max=math.pi - self.eps)
        pos = theta_m.cos()

        # g_additive
        g_add = self.config.margin + (self.config.margin * margin_scaler)
        pos = pos - g_add

        return pos


class SimCLRMarginsLoss(nn.Module):
    """
    SimCLR Margins loss.

    Attributes:
        config (SimCLRMarginsLossConfig): Loss configuration.
        loss_fn (LossFunction): Loss function.
    """

    _LOSS_FUNCTIONS = {
        SimCLRMarginsLossEnum.NTXENT: SimCLRNTXent,
        SimCLRMarginsLossEnum.NTXENT_SPHEREFACE: SimCLRNTXentSphereFace,
        SimCLRMarginsLossEnum.NTXENT_COSFACE: SimCLRNTXentCosFace,
        SimCLRMarginsLossEnum.NTXENT_ARCFACE: SimCLRNTXentArcFace,
        SimCLRMarginsLossEnum.NTXENT_CURRICULARFACE: SimCLRNTXentCurricularFace,
        SimCLRMarginsLossEnum.NTXENT_MAGFACE: SimCLRNTXentMagFace,
        SimCLRMarginsLossEnum.NTXENT_ADAFACE: SimCLRNTXentAdaFace,
    }

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize a SimCLR Margins loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None

        Raises:
            NotImplementedError: If an unsupported loss function is specified.
        """
        super().__init__()

        self.config = config

        self.loss_fn = self._LOSS_FUNCTIONS[config.loss](config)

    def update_margin(self, epoch: int, max_epochs: int) -> float:
        """
        Update margin value for SphereFace, CosFace and ArcFace.

        Args:
            epoch (int): Current training epoch.
            max_epochs (int): Maximum number of epochs.

        Returns:
            int: Margin value.
        """
        if not self.config.margin_scheduler:
            return self.config.margin

        if epoch > (max_epochs // 2):
            margin = self.config.margin
        else:
            margin = (
                self.config.margin
                - self.config.margin
                * (math.cos(math.pi * epoch / (max_epochs // 2)) + 1)
                / 2
            )

        self.loss_fn.margin = margin
        return margin

    def forward(self, Z_1: T, Z_2: T, ssps_assignments: Optional[T] = None) -> T:
        """
        Compute loss.

        Args:
            Z_1 (T): First embeddings tensor.
            Z_2 (T): Second embeddings tensor.
            ssps_assignments (Optional[T]): SSPS assignments to prevent false negatives.

        Returns:
            T: Loss tensor.
        """
        Z = torch.stack((Z_1, Z_2), dim=1)
        # Z shape: (N, V, C)

        loss = self.loss_fn(
            Z,
            Z,
            discard_identity=True,
            ssps_assignments=ssps_assignments,
        )

        return loss
