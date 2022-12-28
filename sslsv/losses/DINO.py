import torch
from torch import nn
import torch.nn.functional as F


class DINOLoss(nn.Module):

    def __init__(
        self,
        nb_prototypes,
        student_temp,
        teacher_temp,
        teacher_temp_warmup,
        teacher_temp_warmup_epochs,
        center_momentum=0.9,
        nb_local_views=4,
        nb_global_views=2
    ):
        super().__init__()

        self.epoch = 0

        self.nb_local_views = nb_local_views
        self.nb_global_views = nb_global_views

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
        # Student
        student_out = S / self.student_temp
        student_out = student_out.chunk(self.nb_local_views)

        # Teacher
        teacher_out = (T - self.center) / self._get_teacher_temp()
        teacher_out = F.softmax(teacher_out, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nb_global_views)

        loss = 0
        nb_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq: continue
                loss += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                nb_loss_terms += 1

        loss /= nb_loss_terms
        self.update_center(T)
        return loss

    @torch.no_grad()
    def update_center(self, T):
        new_center = torch.mean(T, dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum +
            new_center * (1 - self.center_momentum)
        )