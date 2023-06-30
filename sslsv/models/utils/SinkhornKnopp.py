# Adapted from https://github.com/facebookresearch/swav

import torch
from torch import nn

import torch.distributed as dist
from sslsv.utils.distributed import is_dist_initialized, get_world_size


class SinkhornKnopp(nn.Module):

    def __init__(self, nb_iters=3, epsilon=0.05):
        super().__init__()

        self.nb_iters = nb_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, Q):
        B, K = Q.size()
        B *= get_world_size()

        Q = torch.exp(Q / self.epsilon).T

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if is_dist_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(self.nb_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_rows = torch.sum(Q, dim=1, keepdim=True)
            if is_dist_initialized():
                dist.all_reduce(sum_rows)
            Q /= sum_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment

        return Q.T
