import torch
from torch import nn
import torch.nn.functional as F

from .InfoNCELoss import InfoNCELoss


class CPCLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.infonce = InfoNCELoss(temperature=1.0)

    def forward(self, Y_future_preds, Y_future):
        # Shape: (N, encoded_dim, nb_t_to_predict)

        nb_t_to_predict = Y_future.size(2)

        loss = 0
        for t in range(nb_t_to_predict):
            loss += self.infonce(
                Y_future[:, :, t].contiguous(),
                Y_future_preds[:, :, t].contiguous()
            )

        loss /= nb_t_to_predict

        return loss