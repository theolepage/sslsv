import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import math

import torch.distributed as dist

from sslsv.utils.distributed import is_dist_initialized, get_world_size, gather


class DINOSK2Loss(nn.Module):
    """
    DINOSK2 loss.

    Attributes:
        global_count (float): Number of global views.
        local_count (float): Number of local views.
        student_temp (float): Student temperature.
        teacher_temp (float): Teacher temperature.
        memax_weight (float): Weight for Me-Max regularization.
        koleo_weight (float): Weight for Koleo regularization.
    """

    def __init__(
        self,
        global_count: int,
        local_count: int,
        student_temp: float,
        teacher_temp: float,
        memax_weight: float,
        koleo_weight: float,
    ):
        """
        Initialize a DINOSK2 loss.

        Args:
            global_count (int): Number of global views.
            local_count (int): Number of local views.
            student_temp (float): Temperature value for the student.
            teacher_temp (float): Temperature value for the teacher.
            memax_weight (float): Weight for Me-Max regularization.
            koleo_weight (float): Weight for Koleo regularization.

        Returns:
            None
        """
        super().__init__()

        self.global_count = global_count
        self.local_count = local_count

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        self.memax_weight = memax_weight
        self.koleo_weight = koleo_weight

    def forward(self, student: T, teacher: T, Y: T) -> T:
        """
        Compute loss.

        Args:
            student (T): Student embeddings tensor.
            teacher (T): Teacher embeddings tensor.
            Y (T): Student representations tensor.

        Returns:
            T: Loss tensor.
        """
        student = F.softmax(student / self.student_temp, dim=-1)

        with torch.no_grad():
            teacher = F.softmax(teacher / self.teacher_temp, dim=-1)
            teacher = self.sk(teacher)
            teacher = teacher.repeat(self.local_count, 1).detach()

        loss = torch.mean(torch.sum(-teacher * torch.log(student), dim=-1))

        # Me-Max regularization
        student_avg = torch.mean(gather(student), dim=0)
        memax_loss = -torch.sum(-student_avg * torch.log(student_avg)) + math.log(len(student_avg))

        # Koleo regularization
        koleo_loss = torch.stack([self.koleo(v) for v in Y.chunk(self.local_count)]).sum()

        metrics = {
            # `loss_ce` and `loss_koleo` are directly comparable to SDPN's
            # `train_ploss` and `train_ke_loss`.
            "train/loss_ce": loss,
            "train/loss_memax": memax_loss,
            "train/loss_koleo": koleo_loss,
        }

        loss = loss + self.memax_weight * memax_loss + self.koleo_weight * koleo_loss

        return loss, metrics

    @torch.no_grad()
    def sk(self, Q: T, nb_iters: int = 3) -> T:
        """
        Run Sinkhorn-Knopp algorithm.

        Args:
            Q (T): Input tensor. Shape: (B, K).

        Returns:
            T: Output tensor. Shape: (B, K).
        """
        B, K = Q.size()
        B *= get_world_size()

        Q = Q.T

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if is_dist_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(nb_iters):
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

    def koleo(self, student: T, eps: float = 1e-8) -> T:
        """
        Kozachenko-Leonenko loss.

        Args:
            student (T): Student embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        student = F.normalize(student, eps=eps, p=2, dim=-1)

        sim = student @ student.T
        sim.fill_diagonal_(-1)
        idx = torch.max(sim, dim=1)[1]

        pdist = nn.PairwiseDistance(2, eps=eps)
        distances = pdist(student, student[idx])

        loss = -torch.log(distances + eps).mean()
        return loss
