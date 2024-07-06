from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F
from torch import Tensor as T

import torch.distributed as dist
from sslsv.utils.distributed import (
    is_dist_initialized,
    is_main_process,
    gather,
)


class KMeans:
    """
    K-Means algorithm.

    Adapted from https://github.com/facebookresearch/swav/blob/main/main_deepclusterv2.py.

    Attributes:
        nb_prototypes (int): Number of prototypes.
        nb_iters (int): Number of iterations.
        dataset_size (int): Size of the dataset.
        verbose (bool): Wether to show a progress bar for iterations.
        batch_size (int): Batch size for matrix multiplication.
    """

    def __init__(
        self,
        nb_prototypes: int,
        nb_iters: int,
        dataset_size: int,
        verbose: bool = False,
        batch_size: int = 1000,
    ):
        """
        Initialize a K-Means object.

        Args:
            nb_prototypes (Sequence[int]): Number of prototypes.
            nb_iters (int): Number of iterations.
            dataset_size (int): Size of the dataset.
            verbose (bool): Wether to show a progress bar for iterations.
            batch_size (int): Batch size for matrix multiplication.

        Returns:
            None
        """
        self.nb_prototypes = nb_prototypes
        self.nb_iters = nb_iters
        self.dataset_size = dataset_size
        self.verbose = verbose
        self.batch_size = batch_size

    @staticmethod
    def get_indices_sparse(data: np.ndarray) -> List[np.ndarray]:
        data[data == -1] = data.max() + 1  # Fix when assignments contains -1

        cols = np.arange(data.size)
        M = csr_matrix(
            (cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size)
        )
        return [np.unravel_index(row.data, data.shape) for row in M]

    @torch.no_grad()
    def run(
        self,
        local_memory_index: T,
        local_memory_embeddings: T,
    ) -> Tuple[T, List[T]]:
        """
        Run the K-means clustering algorithm.

        Args:
            local_memory_index (T): Tensor of training indexes.
            local_memory_embeddings (T): Tensor of training embeddings.

        Returns:
            Tuple[T, T, T]: Assignments, centroids and similarities tensors.
        """
        N, D = local_memory_embeddings.size()
        device = local_memory_embeddings.device

        K = self.nb_prototypes
        assignments = -1 * torch.ones(
            self.dataset_size,
            dtype=torch.long,
            device=device,
        )
        similarities = torch.zeros(
            self.dataset_size,
            device=device,
        )

        # Init centroids with randomly selected embeddings
        centroids = torch.empty(K, D).to(device, non_blocking=True)
        if is_main_process():
            random_idx = torch.randperm(N)[:K]
            assert len(random_idx) == K
            centroids = local_memory_embeddings[random_idx]
        if is_dist_initialized():
            dist.broadcast(centroids, 0)

        # Run k-means algorithm
        for n_iter in tqdm(
            range(self.nb_iters + 1),
            disable=not self.verbose or not is_main_process(),
            desc="K-Means",
        ):
            # E step
            local_memory_embeddings_batched = local_memory_embeddings.view(
                N // self.batch_size, self.batch_size, -1
            )
            local_matches = [
                (batch @ centroids.T).max(dim=1)
                for batch in local_memory_embeddings_batched
            ]

            local_assignments = torch.cat([batch.indices for batch in local_matches])
            local_similarities = torch.cat([batch.values for batch in local_matches])

            if n_iter == self.nb_iters:
                break

            # M step

            # 1. Determine embeddings indexes belonging to each centroid
            where_helper = self.get_indices_sparse(local_assignments.cpu().numpy())

            # 2. Compute the mean of the embeddings for each centroid
            counts = torch.zeros(K).to(device, non_blocking=True).int()
            emb_sums = torch.zeros(K, D).to(device, non_blocking=True)
            for k in range(K):
                idx = where_helper[k][0]
                if len(idx) > 0:
                    emb_sums[k] = torch.sum(local_memory_embeddings[idx], dim=0)
                    counts[k] = len(idx)
            if is_dist_initialized():
                dist.all_reduce(counts)
                dist.all_reduce(emb_sums)

            # 3. Adjust centroids positions
            mask = counts > 0
            centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

            centroids = F.normalize(centroids, dim=1, p=2)

        indexes_all = gather(local_memory_index)
        assignments_all = gather(local_assignments)
        similarities_all = gather(local_similarities)

        assignments[indexes_all] = assignments_all
        similarities[indexes_all] = similarities_all

        return assignments, centroids, similarities
