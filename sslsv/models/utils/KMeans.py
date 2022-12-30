import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F


class KMeans:

    def __init__(self, nb_prototypes, nb_iters, nb_views):
        self.nb_prototypes = nb_prototypes
        self.nb_iters = nb_iters
        self.nb_views = nb_views

    @staticmethod
    def get_centroid_to_emb_idx(assignments):
        assignments = assignments.cpu().numpy()
        cols = np.arange(assignments.size)
        M = csr_matrix(
            (cols, (assignments.ravel(), cols)),
            shape=(int(assignments.max()) + 1, assignments.size)
        )
        return [np.unravel_index(row.data, assignments.shape) for row in M]

    def run_iter(self, embeddings, centroids_, K, D):
        # E step
        assignments_ = (embeddings @ centroids_.T).max(dim=1)[1]

        # M step

        # 1. Determine embeddings indexes belonging to each centroid
        centroid_to_emb_idx = self.get_centroid_to_emb_idx(assignments_)

        # 2. Compute the mean of the embeddings for each centroid
        counts = torch.zeros(K).to(embeddings.device, non_blocking=True).int()
        emb_sums = torch.zeros(K, D).to(embeddings.device, non_blocking=True)
        for k in range(K):
            idx = centroid_to_emb_idx[k][0]
            if len(idx) > 0:
                emb_sums[k] = torch.sum(embeddings[idx], dim=0)
                counts[k] = len(idx)

        # 3. Adjust centroids positions
        mask = counts > 0
        centroids_[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

        centroids_ = F.normalize(centroids_, dim=1, p=2)

        return centroids_

    def run(self, indexes, embeddings):
        V, N, D = embeddings.size()
        current_view = 0

        assignments = -1 * torch.ones(
            (len(self.nb_prototypes), N),
            dtype=torch.long
        )
        centroids = []

        with torch.no_grad():
            for i_K, K in enumerate(self.nb_prototypes):
                embeddings_ = embeddings[current_view]

                # Init centroids with randomly selected embeddings
                centroids_ = torch.empty(K, D)
                centroids_ = centroids_.to(embeddings_.device, non_blocking=True)
                random_idx = torch.randperm(len(embeddings_))[:K]
                assert len(random_idx) >= K
                centroids_ = embeddings_[random_idx]

                # Run k-means algorithm
                for n_iter in range(self.nb_iters):
                    centroids_ = self.run_iter(embeddings_, centroids_, K, D)

                assignments_ = (embeddings_ @ centroids_.T).max(dim=1)[1]

                centroids.append(centroids_)
                assignments[i_K][indexes] = assignments_

                current_view = (current_view + 1) % self.nb_views

        return assignments, centroids