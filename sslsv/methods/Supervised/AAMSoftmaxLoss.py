import torch
from torch import nn
import torch.nn.functional as F

import math


class AAMSoftmaxLoss(nn.Module):

    def __init__(self, m=0.2, s=30):
        super().__init__()

        self.m = m
        self.s = s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, Z, labels):
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