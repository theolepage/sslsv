import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class ARPLLoss(nn.Module):
    """
    ARPL (Adversarial Reciprocal Points Learning) loss.

    Paper:
        Adversarial Reciprocal Points Learning for Open Set Recognition
        *Guangyao Chen, Peixi Peng, Xiangqian Wang, Yonghong Tian*
        IEEE TPAMI 2022
        https://arxiv.org/abs/2103.00953

    Attributes:
        weight_pl (float): Weight for regularization term.
        temp (float): Temperature value.
        centers (nn.Parameter): Centers parameter.
        radius (nn.Parameter): Radius parameter.
        margin_loss (nn.MarginRankingLoss): Margin rank loss function.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        weight_pl: float = 0.1,
        temp: float = 1.0,
    ):
        """
        Initialize an ARPL loss.

        Args:
            feat_dim (int): Input dimension.
            num_classes (int): Number of classes.
            weight_pl (float): Weight for regularization term. Defaults to 0.1.
            temp (float): Temperature value. Defaults to 1.0.

        Returns:
            None
        """
        super().__init__()

        self.weight_pl = weight_pl
        self.temp = temp

        self.centers = nn.Parameter(0.1 * torch.randn(num_classes, feat_dim))

        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)

        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def _dist_dot(self, features: T, centers: T) -> T:
        """
        Compute cosine similarity.

        Args:
            features (T): Input tensor.
            centers (T): Centers tensor.

        Returns:
            T: Output tensor.
        """
        dist = features @ centers.T
        return dist

    def _dist_l2(self, features: T, centers: T) -> T:
        """
        Compute L2 distance.

        Args:
            features (T): Input tensor.
            centers (T): Centers tensor.

        Returns:
            T: Output tensor.
        """
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        c_2 = torch.sum(torch.pow(centers, 2), dim=1, keepdim=True)
        dist = f_2 - 2 * (features @ centers.T) + c_2.T
        dist = dist / features.size(1)

        return dist

    def forward(self, Z: T, labels: T) -> T:
        """
        Compute loss.

        Args:
            Z (T): Input tensor.
            labels (T): Target labels tensor.

        Returns:
            T: Loss tensor.
        """
        logits = self._dist_l2(Z, self.centers) - self._dist_dot(Z, self.centers)
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.centers[labels, :]
        _dis_known = (Z - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return loss
