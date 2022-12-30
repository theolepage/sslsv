# Adapted from https://github.com/facebookresearch/swav

import torch
from torch import nn


class SinkhornKnopp(nn.Module):

    def __init__(self, nb_iters=3, epsilon=0.05):
        super().__init__()

        self.nb_iters = nb_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, Q):
        B, K = Q.size()

        Q = torch.exp(Q / self.epsilon).T

        # make the matrix sums to 1
        Q /= torch.sum(Q)

        for _ in range(self.nb_iters):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment

        return Q.T
