from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import math


class MoCoMarginsLossEnum(Enum):
    """
    Enumeration representing loss options for MoCo Margins method.

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
class MoCoMarginsLossConfig:
    """
    MoCo Margins loss configuration.

    Attributes:
        loss (MoCoMarginsLossEnum): Type of loss function.
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

    loss: MoCoMarginsLossEnum = MoCoMarginsLossEnum.NTXENT

    scale: float = 30

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


class MoCoNTXent(nn.Module):
    """
    NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss.

    Attributes:
        config (MoCoMarginsLossConfig): Loss configuration.
    """

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__()

        self.config = config

        self.additional_loss = 0

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

    def forward(
        self,
        query: T,
        key: T,
        queue: T,
        current_labels: Optional[T] = None,
        queue_labels: Optional[T] = None,
    ) -> T:
        """
        Compute loss.

        Args:
            query (T): Query tensor.
            key (T): Key tensor.
            queue (T): Queue tensor.
            current_labels (Optional[T]): Labels tensor from the query/key.
            queue_labels (Optional[T]): Labels tensor from the queue.

        Returns:
            T: Loss tensor.
        """
        N, _ = query.size()

        pos = torch.einsum("nc,nc->n", (query, key)).unsqueeze(-1)
        neg = torch.einsum("nc,ck->nk", (query, queue))

        # Prevent class collisions using labels
        if current_labels is not None and queue_labels is not None:
            mask = current_labels.unsqueeze(1) == queue_labels.unsqueeze(0)
            neg[mask] = 0

        # Compute norms of A
        query_norms = torch.norm(query, p=2, dim=1, keepdim=True)  # (N, 1)

        logits = torch.cat(
            (
                self._process_pos(pos, norms=query_norms),
                self._process_neg(neg),
            ),
            dim=1,
        )
        logits = self._process_logits(logits)

        labels = torch.zeros(N, device=query.device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)
        loss += self.additional_loss

        return loss


class MoCoNTXentSphereFace(MoCoNTXent):
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

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-SphereFace loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

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


class MoCoNTXentCosFace(MoCoNTXent):
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

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-CosFace loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

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


class MoCoNTXentArcFace(MoCoNTXentCosFace):
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

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-ArcFace loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

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
        return MoCoNTXentArcFace._add_aam(pos, self.margin)


class MoCoNTXentCurricularFace(MoCoNTXent):
    """
    NT-Xent-CurricularFace loss.

    Paper:
        CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
        *Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang*
        CVPR 2020
        https://arxiv.org/abs/2004.00288
    """

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-CurricularFace loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

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


class MoCoNTXentMagFace(MoCoNTXent):
    """
    NT-Xent-MagFace loss.

    Paper:
        MagFace: A Universal Representation for Face Recognition and Quality Assessment
        *Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou*
        CVPR 2021
        https://arxiv.org/abs/2103.06627
    """

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-MagFace loss.

        Args:
            config (MoCo): MoCoMarginsLossConfig.

        Returns:
            None
        """
        super().__init__(config)

    def _m(self, x_norm: T) -> T:
        """
        Compute margin values.

        Args:
            x_norm (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        return (self.config.u_margin - self.config.l_margin) / (
            self.config.u_a - self.config.l_a
        ) * (x_norm - self.config.l_a) + self.config.l_margin

    def _g(self, x_norm: T) -> T:
        """
        Compute regularization loss term.

        Args:
            x_norm (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        g = 1 / (self.config.u_a**2) * x_norm + 1 / x_norm
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
        norms = torch.clip(norms, min=self.config.l_a, max=self.config.u_a)

        ada_margins = self._m(norms)

        self.additional_loss = self.config.lambda_g * self._g(norms)

        return MoCoNTXentArcFace._add_aam(pos, ada_margins)


class MoCoNTXentAdaFace(MoCoNTXent):
    """
    NT-Xent-AdaFace loss.

    Paper:
        AdaFace: Quality Adaptive Margin for Face Recognition
        *Minchul Kim, Anil K. Jain, Xiaoming Liu*
        CVPR 2022
        https://arxiv.org/abs/2204.00964
    """

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize an NT-Xent-AdaFace loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

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


class MoCoMarginsLoss(nn.Module):
    """
    MoCo Margins loss.

    Attributes:
        config (MoCoMarginsLossConfig): Loss configuration.
        loss_fn (LossFunction): Loss function.
    """

    _LOSS_FUNCTIONS = {
        MoCoMarginsLossEnum.NTXENT: MoCoNTXent,
        MoCoMarginsLossEnum.NTXENT_SPHEREFACE: MoCoNTXentSphereFace,
        MoCoMarginsLossEnum.NTXENT_COSFACE: MoCoNTXentCosFace,
        MoCoMarginsLossEnum.NTXENT_ARCFACE: MoCoNTXentArcFace,
        MoCoMarginsLossEnum.NTXENT_CURRICULARFACE: MoCoNTXentCurricularFace,
        MoCoMarginsLossEnum.NTXENT_MAGFACE: MoCoNTXentMagFace,
        MoCoMarginsLossEnum.NTXENT_ADAFACE: MoCoNTXentAdaFace,
    }

    def __init__(self, config: MoCoMarginsLossConfig):
        """
        Initialize a MoCo Margins loss.

        Args:
            config (MoCoMarginsLossConfig): Loss configuration.

        Returns:
            None

        Raises:
            NotImplementedError: If an unsupported loss function is specified.
        """
        super().__init__()

        self.config = config

        self.loss_fn = self._LOSS_FUNCTIONS[config.loss](config)

    def update_margin(self, epoch, max_epochs):
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

    def forward(
        self,
        query: T,
        key: T,
        queue: T,
        current_labels: Optional[T] = None,
        queue_labels: Optional[T] = None,
    ) -> T:
        """
        Compute loss.

        Args:
            query (T): Query tensor.
            key (T): Key tensor.
            queue (T): Queue tensor.
            current_labels (Optional[T]): Labels tensor from the query/key.
            queue_labels (Optional[T]): Labels tensor from the queue.

        Returns:
            T: Loss tensor.
        """
        loss = self.loss_fn(query, key, queue, current_labels, queue_labels)

        return loss
