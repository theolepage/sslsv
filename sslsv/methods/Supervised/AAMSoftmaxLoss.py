from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import math


class AAMSoftmaxLoss(nn.Module):
    """
    AAM-Softmax (Additive Angular Margin Softmax) loss.

    Paper:
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        *Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou*
        CVPR 2019
        https://arxiv.org/abs/1801.07698

    Attributes:
        m (float): Margin value.
        s (float): Scale value.
        cos_m (float): Cosine of the margin value.
        sin_m (float): Sine of the margin value.
        th (float): Cosine of pi minus the margin value.
        mm (float): Sine of pi minus the margin value times the margin value.
    """

    def __init__(self, m: float = 0.2, s: float = 30):
        """
        Initialize an AAMSoftmax loss.

        Args:
            m (float): Margin value. Defaults to 0.2.
            s (float): Scale value. Defaults to 30.

        Returns:
            None
        """
        super().__init__()

        self.m = m
        self.s = s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, Z: T, labels: T) -> Tuple[T, T]:
        """
        Compute loss.

        Args:
            Z (T): Input tensor. Shape: (N, C).
            labels (T): Target labels tensor. Shape: (N,).

        Returns:
            Tuple[T, T]: Loss and accuracy tensors.
        """
        sine = torch.sqrt((1.0 - torch.mul(Z, Z)).clamp(0, 1))
        phi = Z * self.cos_m - sine * self.sin_m
        phi = torch.where((Z - self.th) > 0, phi, Z - self.mm)

        one_hot = torch.zeros_like(Z)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * Z)
        output = output * self.s

        loss = F.cross_entropy(output, labels)

        # Determine accuracy
        N = labels.size(0)
        pred_indices = torch.argmax(output, dim=1)
        accuracy = torch.sum(torch.eq(pred_indices, labels)) / N

        return loss, accuracy
