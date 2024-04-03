import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F

import torch.distributed as dist
from sslsv.utils.distributed import (
    is_dist_initialized,
    is_main_process,
    get_world_size
)


class KMeans:

    def __init__(self, nb_prototypes, nb_iters, dataset_size):
        self.nb_prototypes = nb_prototypes
        self.nb_iters = nb_iters
        self.dataset_size = dataset_size

    @staticmethod
    def get_indices_sparse(data):
        cols = np.arange(data.size)
        M = csr_matrix(
            (cols, (data.ravel(), cols)),
            shape=(int(data.max()) + 1, data.size)
        )
        return [np.unravel_index(row.data, data.shape) for row in M]

    def run(self, local_memory_index, local_memory_embeddings):
        V, N, D = local_memory_embeddings.size()
        device = local_memory_embeddings.device
    
        j = 0

        assignments = -1 * torch.ones(
            (len(self.nb_prototypes), self.dataset_size),
            dtype=torch.long
        )

        centroids_list = []

        with torch.no_grad():
            for i_K, K in enumerate(self.nb_prototypes):
                # Init centroids with randomly selected embeddings
                centroids = torch.empty(K, D).to(device, non_blocking=True)
                if is_main_process():
                    random_idx = torch.randperm(N)[:K]
                    assert len(random_idx) == K
                    centroids = local_memory_embeddings[j][random_idx]
                if is_dist_initialized():
                    dist.broadcast(centroids, 0)

                # Run k-means algorithm
                for n_iter in range(self.nb_iters + 1):
                    # E step
                    local_assignments = torch.mm(
                        local_memory_embeddings[j],
                        centroids.T
                    ).max(dim=1)[1]

                    if n_iter == self.nb_iters:
                        break

                    # M step

                    # 1. Determine embeddings indexes belonging to each centroid
                    where_helper = self.get_indices_sparse(
                        local_assignments.cpu().numpy()
                    )

                    # 2. Compute the mean of the embeddings for each centroid
                    counts = torch.zeros(K).to(device, non_blocking=True).int()
                    emb_sums = torch.zeros(K, D).to(device, non_blocking=True)
                    for k in range(K):
                        idx = where_helper[k][0]
                        if len(idx) > 0:
                            emb_sums[k] = torch.sum(
                                local_memory_embeddings[j][idx],
                                dim=0
                            )
                            counts[k] = len(idx)
                    if is_dist_initialized():
                        dist.all_reduce(counts)
                        dist.all_reduce(emb_sums)

                    # 3. Adjust centroids positions
                    mask = counts > 0
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    centroids = F.normalize(centroids, dim=1, p=2)

                centroids_list.append(centroids)

                if is_dist_initialized():
                    # gather the assignments
                    assignments_all = torch.empty(
                        get_world_size(),
                        local_assignments.size(0),
                        dtype=local_assignments.dtype,
                        device=local_assignments.device,
                    )
                    assignments_all = list(assignments_all.unbind(0))
                    dist_process = dist.all_gather(
                        assignments_all, local_assignments, async_op=True
                    )
                    dist_process.wait()
                    assignments_all = torch.cat(assignments_all).cpu()

                    # gather the indexes
                    indexes_all = torch.empty(
                        get_world_size(),
                        local_memory_index.size(0),
                        dtype=local_memory_index.dtype,
                        device=local_memory_index.device,
                    )
                    indexes_all = list(indexes_all.unbind(0))
                    dist_process = dist.all_gather(
                        indexes_all, local_memory_index, async_op=True
                    )
                    dist_process.wait()
                    indexes_all = torch.cat(indexes_all).cpu()
                else:
                    assignments_all = local_assignments
                    indexes_all = local_memory_index

                assignments[i_K][indexes_all] = assignments_all

                j = (j + 1) % V

        return assignments, centroids_list