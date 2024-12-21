from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor as T

from sslsv.utils.distributed import gather

from .SSPSSamplingMethods import (
    SSPS_KNNSampling,
    SSPS_KMeansSampling,
    SSPS_KMeansReprSampling,
)
from .SSPSConfig import SSPSConfig, SSPSSamplingMethodEnum


class SSPS(nn.Module):
    """
    Self-Supervised Positive Sampling (SSPS).

    Attributes:
        config (SSPSConfig): SSPS configuration.
        enabled (bool): Whether SSPS is enabled.
        enabled_next_epoch (bool): Whether SSPS is enabled at the next epoch.
        step_metrics (Dict[str, float]): Metrics related to SSPS.
        sampling (_SSPS_BaseSampling): SSPS sampling method.
        train_indices_ref (T): Memory queue of reference indices.
        train_embeddings_ref (T): Memory queue of reference embeddings.
        train_indices_pos (T): Memory queue of positive indices.
        train_embeddings_pos (T): Memory queue of positive embeddings.
        train_ref_size (int): Size of the reference queues adjusted by batch size.
        train_pos_size (int): Size of the positive queues adjusted by batch size.
    """

    _SAMPLING_METHODS = {
        SSPSSamplingMethodEnum.KNN: SSPS_KNNSampling,
        SSPSSamplingMethodEnum.KMEANS: SSPS_KMeansSampling,
        SSPSSamplingMethodEnum.KMEANS_REPR: SSPS_KMeansReprSampling,
    }

    def __init__(self, config: SSPSConfig):
        """
        Initialize a SSPS module.

        Args:
            config (SSPSConfig): SSPS configuration.

        Returns:
            None
        """
        super().__init__()

        self.config = config

        self.enabled = False
        self.enabled_next_epoch = False

        self.step_metrics = {}

        self.sampling = self._SAMPLING_METHODS[config.sampling](config)

    def initialize(
        self,
        dataset_size: int,
        batch_size: int,
        ref_embeddings_dim: int,
        pos_embeddings_dim: int,
        device: torch.device,
        nb_pos_embeddings: int = 1,
    ):
        """
        Initialize memory queues for SSPS.

        Args:
            dataset_size (int): Number of samples in the train set.
            batch_size (int): Batch size.
            ref_embeddings_dim (int): Dimension of representations (Q^).
            pos_embeddings_dim (int): Dimension of embeddings (Q').
            device (torch.device): Device on which tensors will be allocated.
            nb_pos_embeddings (int): Number of positives. Defaults to 1.

        Returns:
            None
        """
        self.batch_size = batch_size

        self.sampling.init(device, dataset_size, batch_size)

        self.train_ref_size = dataset_size - (dataset_size % self.batch_size)
        if self.config.sampling == SSPSSamplingMethodEnum.KNN:
            self.train_ref_size = self.config.pos_queue_size - (
                self.config.pos_queue_size % self.batch_size
            )
        self.register_buffer(
            "train_indices_ref",
            -torch.ones(self.train_ref_size, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "train_embeddings_ref",
            torch.randn(self.train_ref_size, ref_embeddings_dim, device=device),
        )

        self.train_pos_size = self.config.pos_queue_size - (
            self.config.pos_queue_size % self.batch_size
        )
        self.register_buffer(
            "train_indices_pos",
            -torch.ones(self.train_pos_size, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "train_embeddings_pos",
            torch.randn(
                nb_pos_embeddings,
                self.train_pos_size,
                pos_embeddings_dim,
                device=device,
            ),
        )

    def set_epoch(self, epoch: int):
        """
        Update enabled and enabled_next_epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            None
        """
        self.enabled = epoch >= self.config.start_epoch
        self.enabled_next_epoch = epoch >= (self.config.start_epoch - 1)

    def prepare_sampling(self):
        """
        Prepare sampling (e.g. perform clustering).

        Returns:
            None
        """
        if self.enabled:
            self.sampling.prepare(self.train_indices_ref, self.train_embeddings_ref)

    def sample(self, indices: T, Y_ref: T):
        """
        Sample pseudo-positives.

        Args:
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.

        Returns:
            None
        """
        if self.enabled:
            self.step_metrics = self.sampling.sample(
                indices,
                Y_ref,
                self.train_indices_ref,
                self.train_embeddings_ref,
                self.train_indices_pos,
                self.train_embeddings_pos,
            )

    def apply(self, i: int, Z: T) -> T:
        """
        Extract and substitute pseudo-positives.

        Args:
            i (int): Index of positive type in positive memory queue.
            Z (T): Positive embeddings.

        Returns:
            T: Pseudo-positives embeddings.
        """
        if self.enabled:
            Z = self.sampling.apply(Z.clone(), self.train_embeddings_pos[i])

        return Z

    def update_buffers(
        self,
        step_rel: int,
        indices: T,
        Y_ref: T,
        embeddings: Tuple[T],
    ):
        """
        Update memory queues.

        Args:
            step_rel (int): Current relative training step.
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.
            embeddings (Tuple[T]): List of embeddings to insert in positive memory queue.

        Returns:
            None
        """
        if self.enabled_next_epoch:
            start_idx = (step_rel * self.batch_size) % self.train_ref_size
            end_idx = start_idx + self.batch_size
            self.train_indices_ref[start_idx:end_idx] = gather(indices)
            self.train_embeddings_ref[start_idx:end_idx] = gather(Y_ref)

            start_idx = (step_rel * self.batch_size) % self.train_pos_size
            end_idx = start_idx + self.batch_size
            self.train_indices_pos[start_idx:end_idx] = gather(indices)
            for i, Z in enumerate(embeddings):
                self.train_embeddings_pos[i, start_idx:end_idx] = gather(Z.detach())
