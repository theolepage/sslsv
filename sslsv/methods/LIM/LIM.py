from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig
from sslsv.utils.distributed import gather

from .LIMLoss import LIMLoss, LIMLossEnum


@dataclass
class LIMConfig(BaseMethodConfig):
    """
    LIM method configuration.

    Attributes:
        loss (LIMLossEnum): Loss function option. Defaults to LIMLossEnum.BCE.
    """

    loss: LIMLossEnum = LIMLossEnum.BCE


class LIM(BaseMethod):
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

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        return Y_1, Y_2

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
            Z (Tuple[T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Y_1, Y_2 = Z

        N, _ = Y_1.size()

        # Determine negatives
        Y_2_all = gather(Y_2)
        neg_idx = torch.randint(0, Y_2_all.size(0), size=(N,))
        Y_R = Y_2_all[neg_idx]

        pos = F.cosine_similarity(Y_1, Y_2, dim=-1)
        neg = F.cosine_similarity(Y_1, Y_R, dim=-1)

        loss = self.loss_fn(pos, neg)

        self.log_step_metrics(
            {
                "train/loss": loss,
            },
        )

        return loss
