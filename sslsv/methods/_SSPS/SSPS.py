from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sslsv.utils.distributed import gather

from sslsv.methods._SSPS.SSPSSamplingMethods import (
    SSPS_KNNSampling,
    SSPS_KMeansSampling,
    SSPS_KMeansReprSampling,
    SSPS_TwoKMeansSampling,
    SSPS_TwoKMeansReprSampling,
)


class SSPSSamplingMethodEnum(Enum):

    KNN = "knn"
    KMEANS = "kmeans"
    KMEANS_REPR = "kmeans-repr"
    TWO_KMEANS = "2-kmeans"
    TWO_KMEANS_REPR = "2-kmeans-repr"


@dataclass
class SSPSConfig:
    """
    Self-Supervised Positive Sampling (SSPS) configuration.

    Attributes:
    """

    start_epoch: int = 0

    sampling: SSPSSamplingMethodEnum = SSPSSamplingMethodEnum.TWO_KMEANS_REPR

    queue_size: Optional[int] = None

    kmeans_nb_prototypes: int = 50000
    kmeans_nb_iters: int = 10

    video_threshold: float = 0.835
    speaker_threshold: float = 0.9

    sampling_window: int = 10
    sampling_prob: str = "exp_decay"  # or "uniform"
    sampling_prob_exp_decay: float = 0.7

    verbose: bool = False


class SSPS(nn.Module):

    _SAMPLING_METHODS = {
        SSPSSamplingMethodEnum.KNN: SSPS_KNNSampling,
        SSPSSamplingMethodEnum.KMEANS: SSPS_KMeansSampling,
        SSPSSamplingMethodEnum.KMEANS_REPR: SSPS_KMeansReprSampling,
        SSPSSamplingMethodEnum.TWO_KMEANS: SSPS_TwoKMeansSampling,
        SSPSSamplingMethodEnum.TWO_KMEANS_REPR: SSPS_TwoKMeansReprSampling,
    }

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.enabled = False
        self.enabled_next_epoch = False

        self.step_metrics = {}

        self.sampling = self._SAMPLING_METHODS[config.sampling](config)

    def initialize(self, dataset_size, batch_size, embeddings_dim, device):
        self.batch_size = batch_size

        self.sampling.init(device, dataset_size)

        train_dataset_size = dataset_size - (dataset_size % self.batch_size)
        self.queue_size = (
            self.config.queue_size if self.config.queue_size else train_dataset_size
        )

        self.register_buffer(
            "queue_indices",
            torch.zeros(self.queue_size, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "queue_embeddings",
            torch.randn(3, self.queue_size, embeddings_dim, device=device),
        )

    def set_epoch(self, epoch: int):
        self.enabled = epoch >= self.config.start_epoch
        self.enabled_next_epoch = epoch >= (self.config.start_epoch - 1)

    def prepare_sampling(self):
        if self.enabled:
            self.sampling.prepare(self.queue_indices, self.queue_embeddings)

    def sample(self, indices, embeddings):
        if self.enabled:
            self.step_metrics = self.sampling.sample(
                indices,
                embeddings,
                self.queue_indices,
                self.queue_embeddings,
            )

    def apply(self, i, Z):
        if self.enabled:
            Z = self.sampling.apply(Z.clone(), self.queue_embeddings[i])

        return Z

    def update_queues(self, step_rel, indices, Z_ssps, Z_1, Z_2):
        if self.enabled_next_epoch:
            start_idx = (step_rel * self.batch_size) % self.queue_size
            end_idx = start_idx + self.batch_size
            self.queue_indices[start_idx:end_idx] = gather(indices)
            self.queue_embeddings[0, start_idx:end_idx] = gather(Z_ssps)
            self.queue_embeddings[1, start_idx:end_idx] = gather(Z_1.detach())
            self.queue_embeddings[2, start_idx:end_idx] = gather(Z_2.detach())
