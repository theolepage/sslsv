from typing import Iterable, Optional

import torch
from torch.utils.data import DistributedSampler

from sslsv.utils.distributed import get_rank, get_world_size

import math


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper for PyTorch samplers to handle DistributedDataParallel.

    Attributes:
        sampler (torch.utils.data.Sampler): Sampler object to handle.
        num_replicas (int): Number of processes in the distributed environment.
        rank (int): Rank of the current process in the distributed environment.
        num_samples (int): Number of samples for current process.
        total_size (int): Total number of samples for all processes.
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Initialize a DistributedSamplerWrapper object.

        Args:
            sampler (torch.utils.data.Sampler): Sampler object to handle.
            world_size (Optional[int]): Number of processes in the distributed environment.
                If None, the world size is determined dynamically.
            rank (Optional[int]): Rank of the current process in the distributed environment.
                If None, the rank is determined dynamically.
        """
        self.sampler = sampler

        self.num_replicas = world_size if world_size else get_world_size()
        self.rank = rank if rank else get_rank()
        self.num_samples = 0

    def __iter__(self) -> Iterable[int]:
        """
        Generate indices.

        Returns:
            Iterable[int]: Iterable of indices.
        """
        indices = list(self.sampler.__iter__())

        self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        # Ensure that each rank receives the same amount of data
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        # Subsample depending on rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices)

    def set_epoch(self, epoch: int):
        """
        Set the current epoch.

        Args:
            epoch (int): Current epoch.

        Returns:
            None
        """
        self.sampler.set_epoch(epoch)
