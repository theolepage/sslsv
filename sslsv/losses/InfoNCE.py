import torch
from torch import nn
import torch.nn.functional as F

class InfoNCE(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        Z_a, Z_b = data

        N, D = Z_a.size()

        # Determine loss
        dot = F.normalize(Z_a, p=2, dim=1) @ F.normalize(Z_b, p=2, dim=1).T
        dot = dot / 0.07
        labels = torch.arange(N, device=Z_a.device)
        loss = self.ce(dot, labels)

        # Determine accuracy
        pred_indices = torch.argmax(F.softmax(dot, dim=1), dim=1)
        preds_acc = torch.eq(pred_indices, labels)
        accuracy = torch.count_nonzero(preds_acc) / N

        return loss, accuracy