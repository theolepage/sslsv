import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import torch.distributed as dist
from sslsv.utils.distributed import is_dist_initialized, get_world_size


class DINOLoss(nn.Module):
    """
    DINO loss.

    Attributes:
        epoch (int): Current training epoch.
        student_temp (float): Student temperature.
        teacher_temp_warmup (T): Teacher temperature during warmup.
        teacher_temp (float): Teacher temperature after warmup.
        center_momentum (float): Momentum coefficient for updating the center.
        center (T): Tensor for centering.
    """

    def __init__(
        self,
        global_count: int,
        local_count: int,
        nb_prototypes: int,
        student_temp: float,
        teacher_temp: float,
        teacher_temp_warmup: float,
        teacher_temp_warmup_epochs: int,
        center_momentum: float = 0.9,
    ):
        """
        Initialize a DINO loss.

        Args:
            global_count (int): Number of global views.
            local_count (int): Number of local views.
            nb_prototypes (int): Head output dimension.
            student_temp (float): Temperature value for the student.
            teacher_temp (float): Temperature value for the teacher.
            teacher_temperature_warmup (float): Initial temperature value for the teacher.
            teacher_temperature_warmup_epochs (int): Number of epochs for the teacher temperature warmup.
            center_momentum (float): Momentum coefficient for updating the center. Defaults to 0.9.

        Returns:
            None
        """
        super().__init__()

        self.epoch = 0

        self.global_count = global_count
        self.local_count = local_count

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
        """
        Get the teacher temperature for the current epoch.

        Returns:
            float: Teacher temperature.
        """
        if self.epoch < len(self.teacher_temp_warmup):
            return self.teacher_temp_warmup[self.epoch]
        return self.teacher_temp

    def forward(self, student: T, teacher: T) -> T:
        """
        Compute loss.

        Args:
            student (T): Student embeddings tensor.
            teacher (T): Teacher embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        student = F.log_softmax(
            student / self.student_temp,
            dim=-1
        ).chunk(self.global_count + self.local_count)

        teacher_ = F.softmax(
            (teacher - self.center) / self._get_teacher_temp(),
            dim=-1
        )

        teacher_entropy = torch.distributions.Categorical(
            teacher_,
            validate_args=False
        ).entropy().mean()

        teacher_std = teacher_.std(dim=0).mean()

        teacher_ = teacher_.detach().chunk(self.global_count)

        loss = 0
        kl_div = 0
        nb_loss_terms = 0
        for j in range(len(teacher_)):
            for i in range(len(student)):
                if i == j:
                    continue

                loss += torch.sum(-teacher_[j] * student[i], dim=-1).mean()
                kl_div += nn.KLDivLoss(reduction="batchmean")(student[i], teacher_[j])
                nb_loss_terms += 1

        loss /= nb_loss_terms
        kl_div /= nb_loss_terms

        self.update_center(teacher)

        metrics = {
            'train/kl_div': kl_div.detach(),
            'train/teacher_std': teacher_std.detach(),
            'train/teacher_h': teacher_entropy.detach(),
        }

        return loss, metrics

    @torch.no_grad()
    def update_center(self, teacher: T):
        """
        Update center.

        Args:
            teacher (T): Teacher embeddings tensor.

        Returns:
            None
        """
        new_center = torch.sum(teacher, dim=0, keepdim=True)

        if is_dist_initialized():
            dist.all_reduce(new_center)
            new_center /= get_world_size()

        new_center /= teacher.size(0)

        m = self.center_momentum
        self.center = self.center * m + new_center * (1 - m)
