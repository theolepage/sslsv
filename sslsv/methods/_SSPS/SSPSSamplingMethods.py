from typing import Dict, List, Tuple
import torch
from torch import Tensor as T

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

from sslsv.methods._SSPS.KMeans import KMeans
from .SSPSConfig import SSPSConfig

from sslsv.utils.distributed import is_main_process, get_world_size


class _SSPS_BaseSampling:
    """
    Base class for SSPS sampling methods.

    Attributes:
        config (SSPSConfig): SSPS configuration.
        verbose (bool): Whether to log status messages and progress bars.
        global_metrics (Dict[str, float]): SSPS global metrics (constant during epoch).
        df_train (pandas.DataFrame): Train set dataframe to compute speaker and video accuracies.
    """

    def __init__(self, config: SSPSConfig):
        """
        Initialize a SSPS sampling method.

        Args:
            config (SSPSConfig): SSPS configuration.

        Returns:
            None
        """
        self.config = config
        self.verbose = config.verbose

        self.df_train = pd.read_csv("data/voxceleb2_train.csv")
        self.df_train["Video"] = [file.split("/")[-2] for file in self.df_train["File"]]

        self.global_metrics = {}

    def init(self, device: torch.device, dataset_size: int, batch_size: int):
        """
        Initialize sampling.

        Args:
            device (torch.device): Device on which tensors will be allocated.
            dataset_size (int): Number of samples in the train set.
            batch_size (int): Batch size.

        Returns:
            None
        """
        pass

    def prepare(self, train_indices_ref: T, train_embeddings_ref: T):
        """
        Prepare sampling (e.g. perform clustering).

        Args:
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.

        Returns:
            None
        """
        pass

    def sample(
        self,
        indices: T,
        Y_ref: T,
        train_indices_ref: T,
        train_embeddings_ref: T,
        train_indices_pos: T,
        train_embeddings_pos: T,
    ) -> Dict[str, float]:
        """
        Sample pseudo-positives indices.

        Args:
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.
            train_indices_pos (T): Memory queue of positive indices.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            Dict[str, floast]: SSPS metrics.
        """
        raise NotImplementedError

    def apply(self, Z: T, train_embeddings_pos: T) -> T:
        """
        Extract and substitute pseudo-positives.

        Args:
            Z (T): Positives embeddings.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            T: Pseudo-positives embeddings.
        """
        raise NotImplementedError

    def _sample(self, array: List[int], fn: str, exp_lambda: float) -> int:
        """
        Sample an element from a list of indices.

        Args:
            array (List[int]): List of indices.
            fn (str): Probability function used for sampling.
            exp_lambda (float): Exponential lambda for sampling.

        Returns:
            int: Sampled index.
        """
        N = len(array)

        if fn == "exp":
            probabilities = exp_lambda * torch.exp(
                -exp_lambda * torch.arange(N).float()
            )
        else:  # fn == 'uniform':
            probabilities = torch.ones(N) / N

        probabilities = probabilities / probabilities.sum()

        res = array[torch.multinomial(probabilities, 1)]
        return res

    @staticmethod
    def _find_train_indices_in_queue(
        ssps_indices: T,
        train_indices: T,
    ) -> Tuple[T, float]:
        """
        Find train indices in queue.

        Args:
            ssps_indices (T): Pseudo-positives indices in train set.
            train_indices (T): Train indices in queue.

        Returns:
            Tuple[T, float]: Pseudo-positives indices in queue and coverage.
        """
        queue_indices = torch.full_like(ssps_indices, -1)

        coverage = 0
        count = 0

        for i, idx in enumerate(ssps_indices):
            if idx == -1:
                continue

            count += 1

            matches = torch.nonzero(train_indices == idx).view(-1)
            if len(matches) == 0:
                continue

            coverage += 1
            queue_indices[i] = matches[-1].item()

        if count != 0:
            coverage /= count

        return queue_indices, coverage

    def _determine_metrics(
        self,
        indices: T,
        ssps_indices: T,
        sampling_metrics: Dict[str, float] = {},
        repr_sampling: bool = False,
    ) -> Dict[str, float]:
        """
        Determine SSPS metrics (speaker_acc, video_acc, coverage).

        Args:
            indices (T): Indices of current batch..
            ssps_indices (T): Pseudo-positives indices in train set.
            sampling_metrics (Dict[str, float]): Metrics related to sampling.
            repr_sampling (bool): Whether to determine speaker and video accuracies.

        Returns:
            Dict[str, float]: SSPS metrics.
        """
        repr_metrics = {}

        if repr_sampling:
            speaker_acc, video_acc = 0, 0
            for i, ssps_i in zip(indices, ssps_indices):
                if ssps_i == -1:
                    continue

                true_sample = self.df_train.iloc[i.item()]
                pred_sample = self.df_train.iloc[ssps_i.item()]

                speaker_acc += int(pred_sample.Speaker == true_sample.Speaker)
                video_acc += int(pred_sample.Video == true_sample.Video)

            repr_metrics = {
                "ssps_speaker_acc": speaker_acc,
                "ssps_video_acc": video_acc,
            }

        count = torch.sum(ssps_indices != -1).item()

        if count != 0:
            for k, v in sampling_metrics.items():
                sampling_metrics[k] = v / count
            for k, v in repr_metrics.items():
                repr_metrics[k] = v / count

        coverage = count / len(indices)

        return {
            **self.global_metrics,
            **sampling_metrics,
            **repr_metrics,
            "ssps_coverage": coverage,
        }


class SSPS_KNNSampling(_SSPS_BaseSampling):
    """
    SSPS Nearest-Neighbors sampling.
    """

    def __init__(self, config: SSPSConfig):
        """
        Initialize a SSPS Nearest-Neighbors sampling.

        Args:
            config (SSPSConfig): SSPS configuration.

        Returns:
            None
        """
        super().__init__(config)

    def sample(
        self,
        indices: T,
        Y_ref: T,
        train_indices_ref: T,
        train_embeddings_ref: T,
        train_indices_pos: T,
        train_embeddings_pos: T,
    ) -> Dict[str, float]:
        """
        Sample pseudo-positives indices.

        Args:
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.
            train_indices_pos (T): Memory queue of positive indices.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            Dict[str, floast]: SSPS metrics.
        """
        assert len(train_indices_ref) == len(train_indices_pos)

        self.ssps_indices = torch.full_like(indices, -1)

        sim = Y_ref @ train_embeddings_ref.T

        sampling_pool = 0

        for i in range(len(indices)):
            samples_sim = sim[i]
            train_indices_ = torch.nonzero(train_indices_pos != indices[i]).view(-1)

            nearby_samples = torch.topk(
                samples_sim[train_indices_],
                self.config.intra_sampling_size,
            )[1]

            nearby_samples = train_indices_pos[train_indices_[nearby_samples]]

            sampling_pool += len(nearby_samples)

            self.ssps_indices[i] = self._sample(
                nearby_samples,
                self.config.intra_sampling_prob_fn,
                self.config.intra_sampling_prob_exp_lambda,
            )

        metrics = self._determine_metrics(
            indices,
            self.ssps_indices,
            {"ssps_sampling_pool": sampling_pool},
            repr_sampling=True,
        )

        self.ssps_indices, coverage = self._find_train_indices_in_queue(
            self.ssps_indices, train_indices_pos
        )
        metrics["ssps_coverage"] = metrics["ssps_coverage"] * coverage

        return metrics

    def apply(self, Z: T, train_embeddings_pos: T) -> T:
        """
        Extract and substitute pseudo-positives.

        Args:
            Z (T): Positives embeddings.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            T: Pseudo-positives embeddings.
        """
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = train_embeddings_pos[self.ssps_indices[ssps_mask]].clone()
        return Z


class SSPS_KMeansSampling(_SSPS_BaseSampling):
    """
    SSPS K-Means Clustering (centroid) sampling.
    """

    def __init__(self, config: SSPSConfig):
        """
        Initialize a SSPS K-Means Clustering (centroid) sampling.

        Args:
            config (SSPSConfig): SSPS configuration.

        Returns:
            None
        """
        super().__init__(config)

    def init(self, device: torch.device, dataset_size: int, batch_size: int):
        """
        Initialize sampling.

        Args:
            device (torch.device): Device on which tensors will be allocated.
            dataset_size (int): Number of samples in the train set.
            batch_size (int): Batch size.

        Returns:
            None
        """
        self.device = device

        self.kmeans = KMeans(
            dataset_size=dataset_size,
            nb_prototypes=self.config.kmeans_nb_prototypes,
            nb_iters=self.config.kmeans_nb_iters,
            verbose=self.verbose,
            batch_size=batch_size // get_world_size(),
        )

    def _run_kmeans(self, train_indices: T, train_embeddings: T):
        """
        Perform K-Means clustering.

        Args:
            train_indices (T): Train indices.
            train_embeddings (T): Train embeddings.

        Returns:
            None
        """
        self.assignments, self.centroids, self.similarities = self.kmeans.run(
            train_indices,
            train_embeddings,
        )

        # torch.save(self.assignments, "assignments.pt")
        # torch.save(self.centroids, "centroids.pt")
        # torch.save(self.similarities, "similarities.pt")
        # self.assignments = torch.load("assignments.pt", map_location=self.device)
        # self.centroids = torch.load("centroids.pt", map_location=self.device)
        # self.similarities = torch.load("similarities.pt", map_location=self.device)

        # Determine clustering metrics
        labels_video = pd.factorize(self.df_train["Video"])[0].tolist()
        labels_speaker = pd.factorize(self.df_train["Speaker"])[0].tolist()
        nmi_video = normalized_mutual_info_score(
            labels_video, self.assignments.cpu().numpy()
        )
        nmi_speaker = normalized_mutual_info_score(
            labels_speaker, self.assignments.cpu().numpy()
        )
        # ari = adjusted_rand_score(labels_video, self.assignments.cpu().numpy())
        self.global_metrics = {
            "ssps_kmeans_nmi_video": nmi_video,
            "ssps_kmeans_nmi_speaker": nmi_speaker,
            # "ssps_kmeans_ari": ari,
        }

    def _create_cluster_to_nearby_clusters(self):
        """
        Create mapping of cluster to nearby clusters for inter-sampling.

        Returns:
            None
        """
        # self.cluster_to_nearby_clusters = torch.load("cluster_to_nearby_clusters.pt")
        # return

        self.cluster_to_nearby_clusters = {}
        for c in tqdm(
            range(len(self.centroids)),
            disable=not self.verbose or not is_main_process(),
            desc="Create cluster_to_nearby_clusters",
        ):
            clusters_sim = (self.centroids[c].unsqueeze(0) @ self.centroids.T).view(-1)

            nearby_clusters = torch.topk(
                clusters_sim,
                self.config.inter_sampling_size + 1,
            )[1]

            if self.config.inter_sampling_size == 0:
                self.cluster_to_nearby_clusters[c] = nearby_clusters.tolist()
            else:
                self.cluster_to_nearby_clusters[c] = nearby_clusters[1:].tolist()

        # torch.save(self.cluster_to_nearby_clusters, "cluster_to_nearby_clusters.pt")

    def prepare(self, train_indices_ref: T, train_embeddings_ref: T):
        """
        Prepare sampling (e.g. perform clustering).

        Args:
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.

        Returns:
            None
        """
        self._run_kmeans(train_indices_ref, train_embeddings_ref)
        self._create_cluster_to_nearby_clusters()

    def sample(
        self,
        indices: T,
        Y_ref: T,
        train_indices_ref: T,
        train_embeddings_ref: T,
        train_indices_pos: T,
        train_embeddings_pos: T,
    ) -> Dict[str, float]:
        """
        Sample pseudo-positives indices.

        Args:
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.
            train_indices_pos (T): Memory queue of positive indices.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            Dict[str, floast]: SSPS metrics.
        """
        self.ssps_indices = torch.full_like(indices, -1)

        inter_sampling_pool = 0

        for i, idx in enumerate(indices):
            cluster = self.assignments[idx].item()
            if cluster == -1:
                continue

            # Sample one nearby cluster
            nearby_clusters = self.cluster_to_nearby_clusters[cluster]
            if len(nearby_clusters) == 0:
                continue

            inter_sampling_pool += len(nearby_clusters)
            cluster_selected = self._sample(
                nearby_clusters,
                self.config.inter_sampling_prob_fn,
                self.config.inter_sampling_prob_exp_lambda,
            )

            self.ssps_indices[i] = cluster_selected

        metrics = self._determine_metrics(
            indices,
            self.ssps_indices,
            # {"ssps_inter_sampling_pool": inter_sampling_pool},
        )

        return metrics

    def apply(self, Z: T, train_embeddings_pos: T) -> T:
        """
        Extract and substitute pseudo-positives.

        Args:
            Z (T): Positives embeddings.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            T: Pseudo-positives embeddings.
        """
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = self.centroids[self.ssps_indices[ssps_mask]].clone()
        return Z


class SSPS_KMeansReprSampling(SSPS_KMeansSampling):
    """
    SSPS K-Means Clustering (representation) sampling.
    """

    def __init__(self, config: SSPSConfig):
        """
        Initialize a SSPS K-Means Clustering (representation) sampling.

        Args:
            config (SSPSConfig): SSPS configuration.

        Returns:
            None
        """
        super().__init__(config)

    def _create_cluster_to_nearby_samples(self):
        """
        Create mapping of cluster to nearby (assigned) samples for intra-sampling.

        Returns:
            None
        """
        # self.cluster_to_nearby_samples = torch.load("cluster_to_nearby_samples.pt")
        # return

        where_helper = KMeans.get_indices_sparse(self.assignments.cpu().numpy())

        self.cluster_to_nearby_samples = {}
        for c in tqdm(
            range(len(self.centroids)),
            disable=not self.verbose or not is_main_process(),
            desc="Create cluster_to_nearby_samples",
        ):
            samples_idx = torch.from_numpy(where_helper[c][0])

            samples_sim = self.similarities[samples_idx]

            nearby_samples = torch.topk(
                samples_sim,
                min(len(samples_sim), self.config.intra_sampling_size),
            )[1]

            self.cluster_to_nearby_samples[c] = samples_idx[nearby_samples].tolist()

        # torch.save(self.cluster_to_nearby_samples, "cluster_to_nearby_samples.pt")

    def prepare(self, train_indices_ref: T, train_embeddings_ref: T):
        """
        Prepare sampling (e.g. perform clustering).

        Args:
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.

        Returns:
            None
        """
        super().prepare(train_indices_ref, train_embeddings_ref)

        # self._create_cluster_to_nearby_samples()

    def sample(
        self,
        indices: T,
        Y_ref: T,
        train_indices_ref: T,
        train_embeddings_ref: T,
        train_indices_pos: T,
        train_embeddings_pos: T,
    ) -> Dict[str, float]:
        """
        Sample pseudo-positives indices.

        Args:
            indices (T): Indices of current batch.
            Y_ref (T): Reference representations (Y_ref) of current batch.
            train_indices_ref (T): Memory queue of reference indices.
            train_embeddings_ref (T): Memory queue of reference embeddings.
            train_indices_pos (T): Memory queue of positive indices.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            Dict[str, floast]: SSPS metrics.
        """
        self.ssps_indices = torch.full_like(indices, -1)

        inter_sampling_pool = 0
        intra_sampling_pool = 0

        for i, idx in enumerate(indices):
            cluster = self.assignments[idx].item()
            if cluster == -1:
                continue

            # Sample one nearby cluster
            nearby_clusters = self.cluster_to_nearby_clusters[cluster]
            if len(nearby_clusters) == 0:
                continue

            inter_sampling_pool += len(nearby_clusters)
            cluster_selected = self._sample(
                nearby_clusters,
                self.config.inter_sampling_prob_fn,
                self.config.inter_sampling_prob_exp_lambda,
            )

            # Sample one sample from selected cluster
            # nearby_samples = self.cluster_to_nearby_samples[cluster_selected]
            nearby_samples = train_indices_pos[
                self.assignments[train_indices_pos] == cluster_selected
            ]
            if len(nearby_samples) == 0:
                continue

            intra_sampling_pool += len(nearby_samples)
            sample_selected = self._sample(
                nearby_samples,
                self.config.intra_sampling_prob_fn,
                self.config.intra_sampling_prob_exp_lambda,
            )

            self.ssps_indices[i] = sample_selected

        metrics = self._determine_metrics(
            indices,
            self.ssps_indices,
            {
                # "ssps_inter_sampling_pool": inter_sampling_pool,
                "ssps_intra_sampling_pool": intra_sampling_pool,
            },
            repr_sampling=True,
        )

        self.ssps_indices, coverage = self._find_train_indices_in_queue(
            self.ssps_indices, train_indices_pos
        )
        metrics["ssps_coverage"] = metrics["ssps_coverage"] * coverage

        return metrics

    def apply(self, Z: T, train_embeddings_pos: T) -> T:
        """
        Extract and substitute pseudo-positives.

        Args:
            Z (T): Positives embeddings.
            train_embeddings_pos (T): Memory queue of positive embeddings.

        Returns:
            T: Pseudo-positives embeddings.
        """
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = train_embeddings_pos[self.ssps_indices[ssps_mask]].clone()
        return Z
