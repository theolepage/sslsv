import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class CPCConfig(BaseModelConfig):

    bidirectional: bool = False
    nb_t_to_predict: int = 12

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

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.bidirectional = config.bidirectional
        self.nb_t_to_predict = config.nb_t_to_predict

        self.aggregator = CPCAggregator(
            config.aggregator_type,
            config.aggregator_nb_layers,
            encoder.encoder_dim,
            config.aggregator_dim
        )

        self.predictor = CPCPredictor(
            config.aggregator_dim,
            encoder.encoder_dim,
            self.nb_t_to_predict
        )

        if self.bidirectional:
            self.aggregator_r = CPCAggregator(
                config.aggregator_type,
                config.aggregator_nb_layers,
                encoder.encoder_dim,
                config.aggregator_dim
            )

            self.predictor_r = CPCPredictor(
                config.aggregator_dim,
                encoder.encoder_dim,
                self.nb_t_to_predict
            )

    def forward(self, X, training=False):
        Y = super().forward(X)
        
        if self.bidirectional:
            Y_r = super().forward(torch.flip(X, dims=(-1,)))
            if not training:
                return torch.cat(
                    (self.aggregator(Y), self.aggregator(Y_r)),
                    dim=-1
                )
            return Y, Y_r

        return self.aggregator(Y) if not training else Y

    def _cpc_loss(self, Y_future_preds, Y_future):
        # Shape: (N, encoded_dim, nb_t_to_predict)
    
        losses = 0
        for t in range(self.nb_t_to_predict):
            dot = Y_future[:, :, t] @ Y_future_preds[:, :, t].T
            log_softmax_dot = torch.nn.functional.log_softmax(dot, dim=-1)
            diag = torch.diagonal(log_softmax_dot)
            losses += diag

        losses /= self.nb_t_to_predict
        loss = -torch.mean(losses)

        return loss

    def compute_loss_(self, Y):
        # Y: (N, encoded_dim, frame_length / 160)

        N, C, L = Y.size()

        # Number of timesteps used for context
        idx = torch.randint(L - self.nb_t_to_predict + 1, size=(1,))

        Y_past   = Y[:, :, :idx]
        Y_future = Y[:, :, idx:idx+self.nb_t_to_predict]
        # (N, encoded_dim, nb_t_to_predict)

        C = self.aggregator(Y_past)
        # C: (N, aggregator_dim)

        Y_future_preds = self.predictor(C)
        # (N, encoded_dim, nb_t_to_predict)
        
        loss = self._cpc_loss(Y_future_preds, Y_future)

        # Determine accuracy only for the last timestep
        accuracy = InfoNCELoss.determine_accuracy(
            Y_future_preds[:, :, -1],
            Y_future[:, :, -1]
        )

        return loss, accuracy

    def compute_loss(self, Y):
        if self.bidirectional: Y, Y_r = Y

        loss, accuracy = self.compute_loss_(Y)
        
        if self.bidirectional:
            loss_r, accuracy_r = self.compute_loss_(Y_r)
            loss = (loss + loss_r) / 2
            accuracy = (accuracy + accuracy_r) / 2

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics