import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist
from sslsv.utils.distributed import is_dist_initialized, get_world_size


class DINOLoss(nn.Module):

    def __init__(
        self,
        nb_prototypes,
        student_temp,
        teacher_temp,
        teacher_temp_warmup,
        teacher_temp_warmup_epochs,
        center_momentum=0.9
    ):
        super().__init__()

        self.epoch = 0

        self.student_temp = student_temp
        self.teacher_temp_warmup = torch.linspace(
            teacher_temp_warmup,
            teacher_temp,
            teacher_temp_warmup_epochs
        )
        self.teacher_temp = teacher_temp

        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, nb_prototypes))

    def _get_teacher_temp(self):
        if self.epoch < len(self.teacher_temp_warmup):
            return self.teacher_temp_warmup[self.epoch]
        return self.teacher_temp

    def forward(self, S, T):
        S = S / self.student_temp
        S = S.chunk(2 + 4)

        T_ = F.softmax((T - self.center) / self._get_teacher_temp(), dim=-1)
        T_ = T_.detach().chunk(2)

        loss = 0
        nb_loss_terms = 0
        for j in range(len(T_)):
            for i in range(len(S)):
                if i == j: continue

                loss += torch.sum(-T_[j] * F.log_softmax(S[i], dim=-1), dim=-1).mean()
                nb_loss_terms += 1

        loss /= nb_loss_terms

        self.update_center(T)

        return loss

    @torch.no_grad()
    def update_center(self, T):
        new_center = torch.sum(T, dim=0, keepdim=True)

        if is_dist_initialized():
            dist.all_reduce(new_center)
            new_center /= get_world_size()

        new_center /= T.size(0)

        self.center = (
            self.center * self.center_momentum +
            new_center * (1 - self.center_momentum)
        )