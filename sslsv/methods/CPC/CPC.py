from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor as T

from dataclasses import dataclass
from enum import Enum

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .CPCLoss import CPCLoss


class AggregatorTypeEnum(Enum):
    """
    Enumeration representing types of aggregators for CPC.

    Attributes:
        GRU (str): Gated Recurrent Unit (GRU).
        LSTM (str): Long short-term memory (LSTM).
    """

    GRU = "gru"
    LSTM = "lstm"


@dataclass
class CPCConfig(BaseMethodConfig):
    """
    CPC method configuration.

    Attributes:
        bidirectional (bool): Whether to predict backward sequence.
        nb_t_to_predict (int): Number of time-steps to predict.
        aggregator_type (AggregatorTypeEnum): Type of aggregator.
        aggregator_nb_layers (int): Number of layers in the aggregator.
        aggregator_dim (int): Dimension of the aggregator.
    """

    bidirectional: bool = False
    nb_t_to_predict: int = 4

    aggregator_type: AggregatorTypeEnum = AggregatorTypeEnum.GRU
    aggregator_nb_layers: int = 1
    aggregator_dim: int = 256


class CPCAggregator(nn.Module):
    """
    Aggregator for CPC.

    Attributes:
        MODULES_SUPPORTED (Dict[AggregatorTypeEnum, nn.Module]): Dictionary mapping aggregator types
            to corresponding PyTorch modules.

        aggregator (nn.Module): Aggregator module.
    """

    MODULES_SUPPORTED = {
        AggregatorTypeEnum.GRU: nn.GRU,
        AggregatorTypeEnum.LSTM: nn.LSTM,
    }

    def __init__(
        self,
        module_type: AggregatorTypeEnum,
        nb_layers: int,
        encoder_dim: int,
        aggregator_dim: int,
    ):
        """
        Initialize a CPC aggregator.

        Args:
            module_type (AggregatorTypeEnum): Aggregator type.
            nb_layers (int): Number of layers in the aggregator module.
            encoder_dim (int): Dimension of the encoder output.
            aggregator_dim (int): Dimension of the aggregator output.

        Returns:
            None
        """
        super().__init__()

        self.aggregator = CPCAggregator.MODULES_SUPPORTED[module_type](
            encoder_dim,
            aggregator_dim,
            nb_layers,
            batch_first=True,
        )

    def forward(self, X: T) -> T:
        """
        Forward pass.

        Args:
            X (T): Input tensor. Shape: (N, D, L).

        Returns:
            T: Output tensor. Shape: (N, D).
        """
        X = X.transpose(1, 2)  # (N, L, C)
        self.aggregator.flatten_parameters()
        Z, _ = self.aggregator(X)
        return Z[:, -1, :]


class CPCPredictor(nn.Module):
    """
    Predictor for CPC.

    Attributes:
        predictors (nn.ModuleList): Predictor modules.
    """

    def __init__(self, aggregator_dim: int, encoder_dim: int, nb_t_to_predict: int):
        """
        Initializes a CPC predictor.

        Args:
            aggregator_dim (int): Dimension of the aggregator output.
            encoder_dim (int): Dimension of the encoder output.
            nb_t_to_predict (int): Number of time-steps to predict.

        Returns:
            None
        """
        super().__init__()

        self.predictors = nn.ModuleList(
            [nn.Linear(aggregator_dim, encoder_dim) for _ in range(nb_t_to_predict)]
        )

    def forward(self, X: T) -> T:
        """
        Forward pass.

        Args:
            X (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        return torch.stack([predictor(X) for predictor in self.predictors], axis=2)


class CPC(BaseMethod):
    """
    CPC (Contrastive Predictive Coding) method.

    Paper:
        Representation Learning with Contrastive Predictive Coding
        *Aaron van den Oord, Yazhe Li, Oriol Vinyals*
        arXiv preprint 2019
        https://arxiv.org/abs/1807.03748

    Attributes:
        bidirectional (bool): Whether to predict backward sequence.
        nb_t_to_predict (int): Number of time-steps to predict.
        aggregator (CPCAggregator): Aggregator module.
        predictor (CPCPredictor): Predictor module.
        aggregator_r (CPCAggregator): Aggregator reversed for bidirectional mode.
        predictor_r (CPCPredictor): Predictor reversed for bidirectional mode.
        loss_fn (CPCLoss): Loss function.
    """

    def __init__(
        self,
        config: CPCConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize the CPC model.

        Args:
            config (CPCConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.bidirectional = config.bidirectional
        self.nb_t_to_predict = config.nb_t_to_predict

        self.aggregator = CPCAggregator(
            config.aggregator_type,
            config.aggregator_nb_layers,
            self.encoder.encoder_dim,
            config.aggregator_dim,
        )

        self.predictor = CPCPredictor(
            config.aggregator_dim,
            self.encoder.encoder_dim,
            self.nb_t_to_predict,
        )

        if self.bidirectional:
            self.aggregator_r = CPCAggregator(
                config.aggregator_type,
                config.aggregator_nb_layers,
                self.encoder.encoder_dim,
                config.aggregator_dim,
            )

            self.predictor_r = CPCPredictor(
                config.aggregator_dim,
                self.encoder.encoder_dim,
                self.nb_t_to_predict,
            )

        self.loss_fn = CPCLoss()

    def forward(
        self,
        X: T,
        training: bool = False,
    ) -> Union[T, Tuple[T, T, Optional[T], Optional[T]]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, Optional[T], Optional[T]]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            Z = self.aggregator(self.encoder(X))

            if self.bidirectional:
                Z_r = self.aggregator_r(self.encoder(torch.flip(X, dims=(-1,))))
                Z = torch.cat((Z, Z_r), dim=-1)

            return Z

        Y_1 = self.encoder(X[:, 0])
        Y_2 = self.encoder(X[:, 1])

        Y_1_r = None
        Y_2_r = None
        if self.bidirectional:
            Y_1_r = self.encoder(torch.flip(X[:, 0], dims=(-1,)))
            Y_2_r = self.encoder(torch.flip(X[:, 1], dims=(-1,)))

        return Y_1, Y_2, Y_1_r, Y_2_r

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [
            {"params": self.aggregator.parameters()},
            {"params": self.predictor.parameters()},
        ]
        if self.bidirectional:
            extra_learnable_params += [
                {"params": self.aggregator_r.parameters()},
                {"params": self.predictor_r.parameters()},
            ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step_(
        self,
        Y_1: T,
        Y_2: T,
        aggregator: CPCAggregator,
        predictor: CPCPredictor,
    ) -> Tuple[T, float]:
        """
        Perform a training step (aux).

        Args:
            Y_1 (T): Embedding tensor of first view.
            Y_2 (T): Embedding tensor of second view.
            aggregator (CPCAggregator): Aggregator module.
            predictor (CPCPredictor): Predictor module.

        Returns:
            Tuple[T, float]: Loss tensor and accuracy.
        """
        # Y: (N, encoded_dim, frame_length)

        Y_past = Y_1[:, :, : -self.nb_t_to_predict]
        Y_future = Y_2[:, :, -self.nb_t_to_predict :]
        # (N, encoded_dim, nb_t_to_predict)

        C = aggregator(Y_past)
        # C: (N, aggregator_dim)

        Y_future_preds = predictor(C)
        # (N, encoded_dim, nb_t_to_predict)

        loss = self.loss_fn(Y_future_preds, Y_future)

        # Determine accuracy only for the last timestep
        accuracy = 0
        # accuracy = InfoNCELoss.determine_accuracy(
        # Y_future[:, :, -1],
        # Y_future_preds[:, :, -1]
        # )

        return loss, accuracy

    def train_step(
        self,
        Z: Tuple[T, T, Optional[T], Optional[T]],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, Optional[T], Optional[T]]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Y_1, Y_2, Y_1_r, Y_2_r = Z

        loss, accuracy = self.train_step_(Y_1, Y_2, self.aggregator, self.predictor)

        if self.bidirectional:
            loss_r, accuracy_r = self.train_step_(
                Y_1_r,
                Y_2_r,
                self.aggregator_r,
                self.predictor_r,
            )
            loss = (loss + loss_r) / 2
            accuracy = (accuracy + accuracy_r) / 2

        self.log_step_metrics(
            {
                "train/loss": loss,
                # 'train/accuracy': accuracy
            },
        )

        return loss
