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
    """
    Supervised method configuration.

    Attributes:
        nb_classes (int): Number of classes.
        speaker_classification (bool): Whether to use a speaker classifier.
    """

    nb_classes: int = 1211
    speaker_classification: bool = True


class Classifier(nn.Module):
    """
    Classifier for supervised method.

    Attributes:
        classifier (nn.Linear): Classifier module.
    """

    def __init__(self, input_dim: int, nb_classes: int):
        """
        Initialize a Classifier module.

        Args:
            input_dim (int): Dimension of the input.
            nb_classes (int): Number of classes.

        Returns:
            None
        """
        super().__init__()

        self.classifier = nn.Linear(input_dim, nb_classes)

    def forward(self, Z: T) -> T:
        """
        Forward pass.

        Args:
            Z (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        return self.classifier(Z)


class SpeakerClassifier(nn.Module):
    """
    Speaker Classifier for supervised method.

    Attributes:
        classifier_weight (nn.Parameter): Classifier parameter.
    """

    def __init__(self, input_dim: int, nb_classes: int):
        """
        Initialize a Speaker Classifier module.

        Args:
            input_dim (int): Dimension of the input.
            nb_classes (int): Number of classes.

        Returns:
            None
        """
        super().__init__()

        self.classifier_weight = nn.Parameter(torch.FloatTensor(nb_classes, input_dim))
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(self, Z: T) -> T:
        """
        Forward pass.

        Args:
            Z (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        return F.normalize(Z) @ F.normalize(self.classifier_weight).T


class Supervised(BaseMethod):
    """
    Supervised method.

    * Classification (Softmax)
    * Speaker classification (AAM-Softmax)

    Attributes:
        classifier (Classifier): Classifier module.
        loss_fn (nn.Module): Loss function.
    """

    def __init__(
        self,
        config: SupervisedConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a Supervised method.

        Args:
            config (SupervisedConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
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
        """
        Forward pass.

        Args:
            X (T): Input tensor
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            T: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        return self.classifier(self.encoder(X))

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
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
        """
        Perform a training step.

        Args:
            Z (T): Embeddings tensor.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        loss = self.loss_fn(Z, labels)

        if self.config.speaker_classification:
            loss, accuracy = loss
        else:
            accuracy = torch.sum(torch.argmax(Z, dim=-1) == labels) / Z.size(0)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/accuracy": accuracy,
            },
        )

        return loss
