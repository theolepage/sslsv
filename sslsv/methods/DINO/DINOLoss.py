import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import torch.distributed as dist
from sslsv.utils.distributed import is_dist_initialized, get_world_size


class DINOLoss(nn.Module):

    def __init__(
        self,
        nb_prototypes: int,
        student_temp: float,
        teacher_temp: float,
        teacher_temp_warmup: float,
        teacher_temp_warmup_epochs: int,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        self.epoch = 0

        self.student_temp = student_temp
        self.teacher_temp_warmup = torch.linspace(
            teacher_temp_warmup,
            teacher_temp,
            teacher_temp_warmup_epochs,
        )
        self.teacher_temp = teacher_temp

        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, nb_prototypes))

    def _get_teacher_temp(self) -> float:
        if self.epoch < len(self.teacher_temp_warmup):
            return self.teacher_temp_warmup[self.epoch]
        return self.teacher_temp

    def forward(self, student: T, teacher: T) -> T:
        student = student / self.student_temp
        student = student.chunk(2 + 4)

        teacher_ = F.softmax((teacher - self.center) / self._get_teacher_temp(), dim=-1)
        teacher_ = teacher_.detach().chunk(2)

        loss = 0
        nb_loss_terms = 0
        for j in range(len(teacher_)):
            for i in range(len(student)):
                if i == j:
                    continue

                loss += torch.sum(
                    -teacher_[j] * F.log_softmax(student[i], dim=-1), dim=-1
                ).mean()
                nb_loss_terms += 1

        loss /= nb_loss_terms

        self.update_center(teacher)

        return loss

    @torch.no_grad()
    def update_center(self, teacher: T):
        new_center = torch.sum(teacher, dim=0, keepdim=True)

        if is_dist_initialized():
            dist.all_reduce(new_center)
            new_center /= get_world_size()

        new_center /= teacher.size(0)

        m = self.center_momentum
        self.center = self.center * m + new_center * (1 - m)
