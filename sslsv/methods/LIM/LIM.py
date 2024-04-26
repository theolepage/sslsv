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

    loss: LIMLossEnum = LIMLossEnum.BCE


class LIM(BaseMethod):

    def __init__(
        self,
        config: LIMConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = LIMLoss(config.loss)

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T]]:
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
            step,
            {
                "train/loss": loss,
            },
        )

        return loss
