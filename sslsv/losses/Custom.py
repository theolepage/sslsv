import torch
from torch import nn
import torch.nn.functional as F

import math
import random

from .VICReg import VICRegLoss


class BaseLoss(nn.Module):

    def __init__(self, logits_fn=lambda A, B : A @ B.T):
        super().__init__()

        self.logits_fn = logits_fn

    def _determine_logits(self, A, B, discard_diag=True):
        N, V_A, D = A.size()
        _, V_B, _ = B.size()

        A = A.transpose(0, 1).reshape(V_A * N, D)
        B = B.transpose(0, 1).reshape(V_B * N, D)

        A = F.normalize(A, p=2, dim=-1)
        B = F.normalize(B, p=2, dim=-1)
        logits = self.logits_fn(A, B)
        # logits: (V_A*N, V_B*N)

        labels = (torch.arange(N).unsqueeze(0) == torch.arange(N).unsqueeze(1))
        labels = labels.repeat(V_A, V_B).float()
        # labels: (V_A*N, V_B*N)
        # For each sample (row)
        #   - a column with 1 is a positive
        #   - a column with 0 is a negative

        # Discard diagonal
        if discard_diag:
            mask = torch.eye(V_A * N, dtype=torch.bool)
            labels = labels[~mask].view(V_A * N, -1)
            logits = logits[~mask].view(V_A * N, -1)

        pos = logits[labels.bool()].view(V_A * N, -1)
        neg = logits[~labels.bool()].view(V_A * N, -1)

        # pos: (V_A*N, 2) -> 2 positives or 1 positive (if discard_diag)
        # neg: (V_A*N, V_B*N-2) -> V_B*N - 2 negatives

        return pos, neg

    def forward(self, Z_1, Z_2):
        raise NotImplementedError


class TripletLoss(BaseLoss):

    def __init__(self, config):
        super().__init__(logits_fn=lambda A, B : torch.cdist(A, B))

        self.margin = config.loss_margin

        if config.loss_learnable_hyperparams:
            self.margin = nn.Parameter(torch.tensor(self.margin))

    def forward(self, Z_1, Z_2):
        pos, neg = self._determine_logits(Z_1, Z_2)

        pos = torch.mean(pos, dim=-1)
        neg = torch.mean(neg, dim=-1)

        loss = torch.mean(F.relu(pos ** 2 - neg ** 2 + self.margin))
        return loss


class ContrastiveLoss(BaseLoss):

    def __init__(self, config):
        super().__init__()

        self.temperature = config.loss_temperature

    def _determine_labels(self, logits, nb_positives):
        labels = torch.zeros(logits.size(), device=logits.device)
        for i in range(nb_positives): labels[:, i] = 1.0
        return labels

    def _process_pos(self, pos):
        return pos

    def _process_neg(self, neg):
        return neg

    def _process_logits(self, logits):
        return logits / self.temperature

    def forward(self, A, B, discard_diag=True):
        pos, neg = self._determine_logits(A, B, discard_diag)

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


class ContrastiveAngularLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.w = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b = nn.Parameter(torch.tensor(config.loss_init_b))

    def _process_logits(self, logits):
        return self.w * logits + self.b


class ContrastiveAngularDistinctiveLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.w_pos = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b_pos = nn.Parameter(torch.tensor(config.loss_init_b))
        self.w_neg = nn.Parameter(torch.tensor(config.loss_init_w))
        self.b_neg = nn.Parameter(torch.tensor(config.loss_init_b))

    def _process_pos(self, pos):
        return self.w_pos * pos + self.b_pos

    def _process_neg(self, neg):
        return self.w_neg * neg + self.b_neg


class AMSoftmaxLoss(ContrastiveLoss):

    def __init__(self, config):
        super().__init__(config)

        self.margin = config.loss_margin
        self.scale = config.loss_scale

        if config.loss_learnable_hyperparams:
            self.margin = nn.Parameter(torch.tensor(self.margin))
            self.scale = nn.Parameter(torch.tensor(self.scale))

    def _process_pos(self, pos):
        return pos - self.margin

    def _process_logits(self, logits):
        return self.scale * logits


class AAMSoftmaxLoss(AMSoftmaxLoss):

    def __init__(self, config):
        super().__init__(config)

    def _process_pos(self, pos):
        sine = torch.sqrt((1.0 - torch.mul(pos, pos)).clamp(0, 1))
        
        phi = pos * math.cos(self.margin) - sine * math.sin(self.margin)
        
        th = math.cos(math.pi - self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        pos = torch.where((pos - th) > 0, phi, pos - mm)

        return pos


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

        self.enable_multi_views = config.enable_multi_views

        self.loss_fn = self._LOSS_METHODS[config.loss_name](config)

        self.vicreg_scale = config.loss_vicreg_scale
        self.vicreg_loss_fn = VICRegLoss(
            config.loss_vicreg_inv_weight,
            config.loss_vicreg_var_weight,
            config.loss_vicreg_cov_weight
        )

    def forward(self, Z):
        # Z shape: (N, V, C)

        GLOBAL_VIEWS = Z[:, :2]
        LOCAL_VIEWS = Z[:, 2:]

        loss = self.loss_fn(GLOBAL_VIEWS, GLOBAL_VIEWS, discard_diag=True)

        # Multi views
        if self.enable_multi_views:
            loss += self.loss_fn(LOCAL_VIEWS, GLOBAL_VIEWS, discard_diag=False)
            loss /= 2

        loss += self.vicreg_scale * self.vicreg_loss_fn(Z[:, 0], Z[:, 1])
        
        return loss