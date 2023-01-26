import torch
from torch import nn
import torch.nn.functional as F

import math
import random

from .VICReg import VICRegLoss


class BaseLoss(nn.Module):

    def __init__(self, logits_fn=lambda Z : Z @ Z.T):
        super().__init__()

        self.logits_fn = logits_fn

    def _determine_logits(self, Z_1, Z_2):
        N, D = Z_1.size()

        Z = torch.cat((Z_1, Z_2), dim=0)
        Z = F.normalize(Z, p=2, dim=1)

        logits = self.logits_fn(Z)
        # logits: (2N, 2N)

        labels = torch.cat((torch.arange(N), torch.arange(N)), dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Discard diagonal (i = j)
        mask = torch.eye(2 * N, dtype=torch.bool)
        labels = labels[~mask].view(2 * N, -1)
        logits = logits[~mask].view(2 * N, -1)

        # labels: (2N, 2N-1)
        # For each sample (row)
        #   - a column with 1 is a positive
        #   - a column with 0 is a negative

        pos = logits[labels.bool()].view(2 * N, -1)
        neg = logits[~labels.bool()].view(2 * N, -1)

        # pos: (2N, 1) -> 1 positive
        # neg: (2N, 2N-2) -> 2N - 2 positives = 2N-2 negatives

        return pos, neg

    def forward(self, Z_1, Z_2):
        raise NotImplementedError


class TripletLoss(BaseLoss):

    def __init__(self, config):
        super().__init__(logits_fn=lambda Z : torch.cdist(Z, Z))

        self.margin = config.loss_margin

        if config.loss_learnable_hyperparams:
            self.margin = nn.Parameter(torch.tensor(self.margin))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        neg = torch.mean(neg, dim=-1)

        return torch.mean(F.relu(pos ** 2 - neg ** 2 + self.margin))


class ContrastiveLoss(BaseLoss):

    def __init__(self, config):
        super().__init__()

        self.temperature = config.loss_temperature

    def _determine_labels(self, logits):
        return torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=logits.device
        )

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        logits = torch.cat((pos, neg), dim=1) / self.temperature
        labels = self._determine_labels(logits)

        return F.cross_entropy(logits, labels)


class ContrastiveAngularLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.w = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b = nn.Parameter(torch.tensor(config.loss_init_b))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        logits = self.w * torch.cat((pos, neg), dim=1) + self.b
        labels = self._determine_labels(logits)

        return F.cross_entropy(logits, labels)


class ContrastiveAngularDistinctiveLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.w_pos = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b_pos = nn.Parameter(torch.tensor(config.loss_init_b))
        self.w_neg = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b_neg = nn.Parameter(torch.tensor(config.loss_init_b))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        pos = self.w_pos * pos + self.b_pos
        neg = self.w_neg * neg + self.b_neg

        logits = torch.cat((pos, neg), dim=1)
        labels = self._determine_labels(logits)

        return F.cross_entropy(logits, labels)


class AMSoftmaxLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.margin = config.loss_margin
        self.scale = config.loss_scale

        if config.loss_learnable_hyperparams:
            self.margin = nn.Parameter(torch.tensor(self.margin))
            self.scale = nn.Parameter(torch.tensor(self.scale))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        pos = pos - self.margin

        logits = torch.cat((pos, neg), dim=1) * self.scale
        labels = self._determine_labels(logits)

        return F.cross_entropy(logits, labels)


class AAMSoftmaxLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.margin = config.loss_margin
        self.scale = config.loss_scale

        if config.loss_learnable_hyperparams:
            self.margin = nn.Parameter(torch.tensor(self.margin))
            self.scale = nn.Parameter(torch.tensor(self.scale))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        sine = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))
        
        phi = pos * math.cos(self.margin) - sine * math.sin(self.margin)
        
        th = math.cos(math.pi - self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        pos = torch.where((pos - th) > 0, phi, pos - mm)

        logits = torch.cat((pos, neg), dim=1) * self.scale
        labels = self._determine_labels(logits)

        return F.cross_entropy(logits, labels)


class CustomLoss(nn.Module):

    _LOSS_METHODS = {
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'angular': ContrastiveAngularLoss,
        'angular_distinctive': ContrastiveAngularDistinctiveLoss,
        'amsoftmax': AMSoftmaxLoss,
        'aamsoftmax': AAMSoftmaxLoss
    }

    def __init__(self, config):
        super().__init__()

        self.loss_fn = self._LOSS_METHODS[config.loss_name](config)

        self.vicreg_scale = config.loss_vicreg_scale
        self.vicreg_loss_fn = VICRegLoss(
            config.loss_vicreg_inv_weight,
            config.loss_vicreg_var_weight,
            config.loss_vicreg_cov_weight
        )

    def forward(self, Z_1, Z_2):
        loss = self.loss_fn(Z_1, Z_2)
        loss += self.vicreg_scale * self.vicreg_loss_fn(Z_1, Z_2)
        return loss