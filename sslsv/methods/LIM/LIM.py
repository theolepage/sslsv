from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig
from sslsv.utils.distributed import gather

from .LIMLoss import LIMLoss, LIMLossEnum


@dataclass
class LIMConfig(BaseSiameseMethodConfig):
    """
    LIM method configuration.

    Attributes:
        loss (LIMLossEnum): Loss function option. Defaults to LIMLossEnum.BCE.
    """

    loss: LIMLossEnum = LIMLossEnum.BCE


class LIM(BaseSiameseMethod):
    """
    LIM (Local Info Max) method.

    Paper:
        Learning Speaker Representations with Mutual Information
        *Mirco Ravanelli, Yoshua Bengio*
        INTERSPEECH 2019
        https://arxiv.org/abs/1812.00271

    Attributes:
        loss_fn (LIMLoss): Loss function.
    """

    def __init__(
        self,
        config: LIMConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a LIM method.

        Args:
            config (LIMConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.loss_fn = LIMLoss(config.loss)

    def train_step(
        self,
        Z: Tuple[T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Z_1, Z_2, _ = Z

        N, _ = Z_1.size()

        # Determine negatives
        Z_2_all = gather(Z_2)
        neg_idx = torch.randint(0, Z_2_all.size(0), size=(N,))
        Z_R = Z_2_all[neg_idx]

        pos = F.cosine_similarity(Z_1, Z_2, dim=-1)
        neg = F.cosine_similarity(Z_1, Z_R, dim=-1)

        loss = self.loss_fn(pos, neg)

        self.log_step_metrics(
            {
                "train/loss": loss,
            },
        )

        return loss
