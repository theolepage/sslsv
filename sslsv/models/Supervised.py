import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.AAMSoftmax import AAMSoftmaxLoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class SupervisedConfig(BaseModelConfig):
    
    nb_speakers: int = 1211


class Classifier(nn.Module):

    def __init__(self, input_dim, nb_speakers):
        super().__init__()

        self.classifier_weight = nn.Parameter(
            torch.FloatTensor(nb_speakers, input_dim)
        )
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(self, Z):
        return F.normalize(Z) @ F.normalize(self.classifier_weight).T


class Supervised(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.config = config

        self.classifier = Classifier(
            self.encoder.encoder_dim,
            config.nb_speakers
        )

        self.loss_fn = AAMSoftmaxLoss()

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        return self.classifier(self.encoder(X))

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.classifier.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(self, Z, labels, step, samples):
        loss, accuracy = self.loss_fn(Z, labels)

        metrics = {
            'train/loss': loss,
            'train/accuracy': accuracy
        }

        return loss, metrics