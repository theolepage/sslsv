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
)


class SSPSSamplingMethodEnum(Enum):

    KNN = "knn"
    KMEANS = "kmeans"
    KMEANS_REPR = "kmeans-repr"


@dataclass
class SSPSConfig:
    """
    Self-Supervised Positive Sampling (SSPS) configuration.

    Attributes:
    """

    start_epoch: int = 100

    sampling: SSPSSamplingMethodEnum = SSPSSamplingMethodEnum.KMEANS_REPR

    pos_queue_size: int = 50000

    kmeans_nb_prototypes: int = 50000
    kmeans_nb_iters: int = 10

    inter_sampling_size: int = 0
    inter_sampling_prob_fn: str = "uniform"  # or "exp_decay"
    inter_sampling_prob_exp_lambda: float = 0.7

    intra_sampling_size: int = 10000000
    intra_sampling_prob_fn: str = "uniform"  # or "exp_decay"
    intra_sampling_prob_exp_lambda: float = 0.7

    verbose: bool = False


class SSPS(nn.Module):

    _SAMPLING_METHODS = {
        SSPSSamplingMethodEnum.KNN: SSPS_KNNSampling,
        SSPSSamplingMethodEnum.KMEANS: SSPS_KMeansSampling,
        SSPSSamplingMethodEnum.KMEANS_REPR: SSPS_KMeansReprSampling,
    }

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.enabled = False
        self.enabled_next_epoch = False

        self.step_metrics = {}

        self.sampling = self._SAMPLING_METHODS[config.sampling](config)

    def initialize(
        self,
        dataset_size,
        batch_size,
        ref_embeddings_dim,
        pos_embeddings_dim,
        device,
        nb_pos_embeddings=1,
    ):
        self.batch_size = batch_size

        self.sampling.init(device, dataset_size, batch_size)

        self.train_buffer_size_ref = dataset_size - (dataset_size % self.batch_size)
        self.register_buffer(
            "train_indices_ref",
            -torch.ones(self.train_buffer_size_ref, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "train_embeddings_ref",
            torch.randn(self.train_buffer_size_ref, ref_embeddings_dim, device=device),
        )

        self.train_buffer_size_pos = self.config.pos_queue_size - (
            self.config.pos_queue_size % self.batch_size
        )
        self.register_buffer(
            "train_indices_pos",
            -torch.ones(self.train_buffer_size_pos, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "train_embeddings_pos",
            torch.randn(
                nb_pos_embeddings,
                self.train_buffer_size_pos,
                pos_embeddings_dim,
                device=device,
            ),
        )

    def set_epoch(self, epoch: int):
        self.enabled = epoch >= self.config.start_epoch
        self.enabled_next_epoch = epoch >= (self.config.start_epoch - 1)

    def prepare_sampling(self):
        if self.enabled:
            self.sampling.prepare(self.train_indices_ref, self.train_embeddings_ref)

    def sample(self, indices, embeddings):
        if self.enabled:
            self.step_metrics = self.sampling.sample(
                indices,
                embeddings,
                self.train_indices_ref,
                self.train_embeddings_ref,
                self.train_indices_pos,
                self.train_embeddings_pos,
            )

    def apply(self, i, Z):
        if self.enabled:
            Z = self.sampling.apply(Z.clone(), self.train_embeddings_pos[i])

        return Z

    def update_buffers(self, step_rel, indices, Z_ssps, embeddings):
        if self.enabled_next_epoch:
            start_idx = step_rel * self.batch_size
            end_idx = start_idx + self.batch_size
            self.train_indices_ref[start_idx:end_idx] = gather(indices)
            self.train_embeddings_ref[start_idx:end_idx] = gather(Z_ssps)

            start_idx = (step_rel * self.batch_size) % self.train_buffer_size_pos
            end_idx = start_idx + self.batch_size
            self.train_indices_pos[start_idx:end_idx] = gather(indices)
            for i, Z in enumerate(embeddings):
                self.train_embeddings_pos[i, start_idx:end_idx] = gather(Z.detach())
