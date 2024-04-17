from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .AAMSoftmaxLoss import AAMSoftmaxLoss


@dataclass
class SupervisedConfig(BaseMethodConfig):

    nb_classes: int = 1211
    speaker_classification: bool = True


class Classifier(nn.Module):

    def __init__(self, input_dim: int, nb_classes: int):
        super().__init__()

        self.classifier = nn.Linear(input_dim, nb_classes)

    def forward(self, Z: T) -> T:
        return self.classifier(Z)


class SpeakerClassifier(nn.Module):

    def __init__(self, input_dim: int, nb_classes: int):
        super().__init__()

        self.classifier_weight = nn.Parameter(torch.FloatTensor(nb_classes, input_dim))
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(self, Z: T) -> T:
        return F.normalize(Z) @ F.normalize(self.classifier_weight).T


class Supervised(BaseMethod):

    def __init__(
        self,
        config: SupervisedConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        classifier_cls = (
            SpeakerClassifier if config.speaker_classification else Classifier
        )
        self.classifier = classifier_cls(self.encoder.encoder_dim, config.nb_classes)

        loss_cls = (
            AAMSoftmaxLoss if config.speaker_classification else nn.CrossEntropyLoss
        )
        self.loss_fn = loss_cls()

    def forward(self, X: T, training: bool = False) -> T:
        if not training:
            return self.encoder(X)

        return self.classifier(self.encoder(X))

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        extra_learnable_params = [{"params": self.classifier.parameters()}]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(
        self,
        Z: T,
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        loss = self.loss_fn(Z, labels)

        if self.config.speaker_classification:
            loss, accuracy = loss
        else:
            accuracy = torch.sum(torch.argmax(Z, dim=-1) == labels) / Z.size(0)

        self.log_step_metrics(
            step,
            {
                "train/loss": loss,
                "train/accuracy": accuracy,
            },
        )

        return loss
