import torch
from torch import nn
import torch.nn.functional as F


class CPCLoss(nn.Module):

    def __init__(self, nb_t_to_predict):
        super().__init__()

        self.nb_t_to_predict = nb_t_to_predict

    def forward(self, Y_future_preds, Y_future):
        # Shape: (N, encoded_dim, nb_t_to_predict)
    
        losses = 0
        for t in range(self.nb_t_to_predict):
            dot = Y_future[:, :, t] @ Y_future_preds[:, :, t].T
            log_softmax_dot = torch.nn.functional.log_softmax(dot, dim=1)
            diag = torch.diagonal(log_softmax_dot)
            losses += diag

        losses /= self.nb_t_to_predict
        loss = -torch.mean(losses)

        return loss