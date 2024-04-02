import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.AAMSoftmax import AAMSoftmaxLoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class SupervisedConfig(BaseModelConfig):
    
    nb_classes: int = 1211
    speaker_classification: bool = True


class Classifier(nn.Module):

    def __init__(self, input_dim, nb_classes):
        super().__init__()

        self.classifier = nn.Linear(input_dim, nb_classes)

    def forward(self, Z):
        return self.classifier(Z)


class SpeakerClassifier(nn.Module):

    def __init__(self, input_dim, nb_classes):
        super().__init__()

        self.classifier_weight = nn.Parameter(
            torch.FloatTensor(nb_classes, input_dim)
        )
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(self, Z):
        return F.normalize(Z) @ F.normalize(self.classifier_weight).T


class Supervised(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.config = config

        classifier_cls = (
            SpeakerClassifier
            if config.speaker_classification
            else Classifier
        )
        self.classifier = classifier_cls(
            self.encoder.encoder_dim,
            config.nb_classes
        )

        loss_cls = (
            AAMSoftmaxLoss
            if config.speaker_classification
            else nn.CrossEntropyLoss
        )
        self.loss_fn = loss_cls()

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        return self.classifier(self.encoder(X))

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.classifier.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(self, Z, labels, step, samples):
        loss = self.loss_fn(Z, labels)

        if self.config.speaker_classification:
            loss, accuracy = loss
            metrics = {
                'train/loss': loss,
                'train/accuracy': accuracy
            }
        else:
            accuracy = torch.sum(torch.argmax(Z, dim=-1) == labels) / Z.size(0)
            metrics = {
                'train/loss': loss,
                'train/accuracy': accuracy
            }

        return loss, metrics