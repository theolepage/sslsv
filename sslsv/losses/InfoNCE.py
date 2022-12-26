import torch
from torch import nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()

        self.temperature = temperature

        self.ce = torch.nn.CrossEntropyLoss()

    @staticmethod
    def dot(Z_a, Z_b):
        return F.normalize(Z_a, p=2, dim=1) @ F.normalize(Z_b, p=2, dim=1).T

    @staticmethod
    def determine_accuracy(Z_a, Z_b):
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, Z_b)
        labels = torch.arange(N, device=dot.device)

        pred_indices = torch.argmax(F.softmax(dot, dim=1), dim=1)
        preds_acc = torch.eq(pred_indices, labels)
        accuracy = torch.count_nonzero(preds_acc) / N
        return accuracy

    def forward(self, Z_a, Z_b):
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, Z_b) / self.temperature
        labels = torch.arange(N, device=dot.device)

        loss = self.ce(dot, labels)
        return loss