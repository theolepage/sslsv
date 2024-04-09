from torch.utils.data import DistributedSampler

from sslsv.utils.distributed import get_rank, get_world_size

import math


class DistributedSamplerWrapper(DistributedSampler):

    def __init__(self, sampler, world_size=None, rank=None):
        self.sampler = sampler

        self.num_replicas = world_size if world_size else get_world_size()
        self.rank = rank if rank else get_rank()
        self.num_samples = 0

    def __iter__(self):
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

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
