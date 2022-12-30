import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.CPC import CPCLoss
from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class CPCConfig(BaseModelConfig):

    bidirectional: bool = False
    nb_t_to_predict: int = 12
    min_nb_t_for_context: int = 100

    aggregator_type: str = 'gru'
    aggregator_nb_layers: int = 1
    aggregator_dim: int = 256


class CPCAggregator(nn.Module):

    MODULES_SUPPORTED = {
        'gru':  nn.GRU,
        'lstm': nn.LSTM
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


class CPC(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.bidirectional = config.bidirectional
        self.nb_t_to_predict = config.nb_t_to_predict
        self.min_nb_t_for_context = config.min_nb_t_for_context

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

        self.loss_fn = CPCLoss(self.nb_t_to_predict)

    def forward(self, X, training=False):
        Y = self.encoder(X)
        Y_r = (
            self.encoder(torch.flip(X, dims=(-1,)))
            if self.bidirectional
            else None
        )
        
        if not training:
            Z = self.aggregator(Y)
            if self.bidirectional:
                Z_r = self.aggregator(Y_r)
                return torch.cat((Z, Z_r), dim=-1)
            return Z

        return Y, Y_r

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

    def train_step_(self, Y):
        # Y: (N, encoded_dim, frame_length / 160)

        N, C, L = Y.size()

        # Number of timesteps used for context
        idx = torch.randint(
            self.min_nb_t_for_context,
            L - self.nb_t_to_predict + 1,
            size=(1,)
        )

        Y_past   = Y[:, :, :idx]
        Y_future = Y[:, :, idx:idx+self.nb_t_to_predict]
        # (N, encoded_dim, nb_t_to_predict)

        C = self.aggregator(Y_past)
        # C: (N, aggregator_dim)

        Y_future_preds = self.predictor(C)
        # (N, encoded_dim, nb_t_to_predict)
        
        loss = self.loss_fn(Y_future_preds, Y_future)

        # Determine accuracy only for the last timestep
        accuracy = InfoNCELoss.determine_accuracy(
            Y_future[:, :, -1],
            Y_future_preds[:, :, -1]
        )

        return loss, accuracy

    def train_step(self, Y, step, samples):
        Y, Y_r = Y

        loss, accuracy = self.train_step_(Y)
        
        if self.bidirectional:
            loss_r, accuracy_r = self.train_step_(Y_r)
            loss = (loss + loss_r) / 2
            accuracy = (accuracy + accuracy_r) / 2

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics