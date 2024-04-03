import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from enum import Enum

from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .CPCLoss import CPCLoss


class AggregatorTypeEnum(Enum):

    GRU = 'gru'
    LSTM = 'lstm'


@dataclass
class CPCConfig(BaseMethodConfig):

    bidirectional: bool = False
    nb_t_to_predict: int = 4

    aggregator_type: AggregatorTypeEnum = AggregatorTypeEnum.GRU
    aggregator_nb_layers: int = 1
    aggregator_dim: int = 256


class CPCAggregator(nn.Module):

    MODULES_SUPPORTED = {
        AggregatorTypeEnum.GRU:  nn.GRU,
        AggregatorTypeEnum.LSTM: nn.LSTM
    }

    def __init__(
        self,
        module_type,
        nb_layers,
        encoder_dim,
        aggregator_dim
    ):
        super().__init__()

        self.aggregator = CPCAggregator.MODULES_SUPPORTED[module_type](
            encoder_dim,
            aggregator_dim,
            nb_layers,
            batch_first=True
        )

    def forward(self, X):
        X = X.transpose(1, 2) # (N, L, C)
        self.aggregator.flatten_parameters()
        Z, _ = self.aggregator(X)
        return Z[:, -1, :]


class CPCPredictor(nn.Module):

    def __init__(self, aggregator_dim, encoder_dim, nb_t_to_predict):
        super().__init__()

        self.predictors = nn.ModuleList([
            nn.Linear(aggregator_dim, encoder_dim)
            for _ in range(nb_t_to_predict)
        ])

    def forward(self, X):
        return torch.stack(
            [predictor(X) for predictor in self.predictors],
            axis=2
        )


class CPC(BaseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.bidirectional = config.bidirectional
        self.nb_t_to_predict = config.nb_t_to_predict

        self.aggregator = CPCAggregator(
            config.aggregator_type,
            config.aggregator_nb_layers,
            self.encoder.encoder_dim,
            config.aggregator_dim
        )

        self.predictor = CPCPredictor(
            config.aggregator_dim,
            self.encoder.encoder_dim,
            self.nb_t_to_predict
        )

        if self.bidirectional:
            self.aggregator_r = CPCAggregator(
                config.aggregator_type,
                config.aggregator_nb_layers,
                self.encoder.encoder_dim,
                config.aggregator_dim
            )

            self.predictor_r = CPCPredictor(
                config.aggregator_dim,
                self.encoder.encoder_dim,
                self.nb_t_to_predict
            )

        self.loss_fn = CPCLoss()

    def forward(self, X, training=False):
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

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.aggregator.parameters()},
            {'params': self.predictor.parameters()}
        ]
        if self.bidirectional:
            extra_learnable_params += [
                {'params': self.aggregator_r.parameters()},
                {'params': self.predictor_r.parameters()}
            ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step_(self, Y_1, Y_2, aggregator, predictor):
        # Y: (N, encoded_dim, frame_length)

        Y_past   = Y_1[:, :, :-self.nb_t_to_predict]
        Y_future = Y_2[:, :, -self.nb_t_to_predict:]
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

    def train_step(self, Y, labels, step, samples):
        Y_1, Y_2, Y_1_r, Y_2_r = Y

        loss, accuracy = self.train_step_(
            Y_1,
            Y_2,
            self.aggregator,
            self.predictor
        )

        if self.bidirectional:
            loss_r, accuracy_r = self.train_step_(
                Y_1_r,
                Y_2_r,
                self.aggregator_r,
                self.predictor_r
            )
            loss = (loss + loss_r) / 2
            accuracy = (accuracy + accuracy_r) / 2

        metrics = {
            'train/loss': loss,
            # 'train/accuracy': accuracy
        }

        return loss, metrics