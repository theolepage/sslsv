import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Sequence

from sslsv.losses.DeepCluster import DeepClusterLoss
from sslsv.models.utils.KMeans import KMeans
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class DeepClusterConfig(BaseModelConfig):

    temperature: float = 0.1

    nb_prototypes: Sequence[int] = field(
        default_factory=lambda: [3000, 3000, 3000]
    )

    kmeans_nb_iters: int = 10

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128


class DeepCluster(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.config = config

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.prototypes = nn.ModuleList([
            nn.Linear(config.projector_output_dim, K, bias=False)
            for K in config.nb_prototypes
        ])
        for layer in self.prototypes:
            for p in layer.parameters(): p.requires_grad = False
            layer.weight.copy_(F.normalize(layer.weight.data.clone(), dim=-1))

        self.kmeans = KMeans(
            nb_prototypes=config.nb_prototypes,
            nb_iters=config.kmeans_nb_iters,
            nb_views=2
        )

        self.loss_fn = DeepClusterLoss(config.temperature)

    def on_train_start(self, trainer):
        self.dataset_size = len(trainer.train_dataloader.dataset)
        self.batch_size = trainer.config.training.batch_size

        self.assignments = -1 * torch.ones(
            (len(self.config.nb_prototypes), self.dataset_size),
            dtype=torch.long,
            device=trainer.device
        )

        self.register_buffer(
            'local_memory_indexes',
            torch.zeros(self.dataset_size).long().to(trainer.device, non_blocking=True)
        )
        self.register_buffer(
            'local_memory_embeddings',
            F.normalize(
                torch.randn(2, self.dataset_size, self.config.projector_output_dim),
                dim=-1,
            ).to(trainer.device, non_blocking=True),
        )

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Z_1 = F.normalize(self.projector(self.encoder(X_1)), dim=-1)
        Z_2 = F.normalize(self.projector(self.encoder(X_2)), dim=-1)

        P_1 = torch.stack([layer(Z_1) for layer in self.prototypes])
        P_2 = torch.stack([layer(Z_2) for layer in self.prototypes])

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch, max_epochs):
        if epoch > 0:
            self.assignments, centroids = self.kmeans.run(
                self.local_memory_indexes,
                self.local_memory_embeddings
            )

            for layer, centroid in zip(self.prototypes, centroids):
                layer.weight.copy_(centroid)

    def train_step(self, Z, step, samples):
        Z_1, Z_2, P_1, P_2 = Z

        preds = torch.stack((P_1, P_2), dim=1)
        assignments = self.assignments[:, samples]

        loss = self.loss_fn(preds, assignments)

        # Store indexes and embeddings for the next clustering step
        start_idx = step * self.batch_size
        end_idx = (step + 1) * self.batch_size
        self.local_memory_indexes[start_idx:end_idx] = samples
        self.local_memory_embeddings[0, start_idx:end_idx] = Z_1.detach()
        self.local_memory_embeddings[1, start_idx:end_idx] = Z_2.detach()

        metrics = {
            'train_loss': loss
        }

        return loss, metrics