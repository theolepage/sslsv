import torch

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from sslsv.methods._SSPS.KMeans import KMeans

from sslsv.utils.distributed import is_main_process, get_rank, get_world_size


class _SSPS_BaseSampling:

    def __init__(self, config):
        self.config = config
        self.verbose = config.verbose

        self.df_train = pd.read_csv("data/voxceleb2_train.csv")
        self.df_train["Video"] = [file.split("/")[-2] for file in self.df_train["File"]]

        self.global_metrics = {}

    def init(self, device, dataset_size, batch_size):
        pass

    def prepare(self, train_indices_ref, train_embeddings_ref):
        pass

    def sample(
        self,
        indices,
        embeddings,
        train_indices_ref,
        train_embeddings_ref,
        train_indices_pos,
        train_embeddings_pos,
    ):
        raise NotImplementedError

    def apply(self, Z, train_embeddings_pos):
        raise NotImplementedError

    def _sample(self, array, fn, exp_lambda):
        N = len(array)

        if fn == "exp_decay":
            probabilities = exp_lambda * torch.exp(
                -exp_lambda * torch.arange(N).float()
            )
        else:  # fn == 'uniform':
            probabilities = torch.ones(N) / N

        probabilities = probabilities / probabilities.sum()

        res = array[torch.multinomial(probabilities, 1)]
        return res

    @staticmethod
    def _find_train_indices_in_queue(ssps_indices, train_indices):
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
        indices,
        ssps_indices,
        sampling_metrics={},
        repr_sampling=False,
    ):
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

    def __init__(self, config):
        super().__init__(config)

        self.config = config

    def sample(
        self,
        indices,
        embeddings,
        train_indices_ref,
        train_embeddings_ref,
        train_indices_pos,
        train_embeddings_pos,
    ):
        assert len(train_indices_ref) == len(train_indices_pos)

        self.ssps_indices = torch.full_like(indices, -1)

        sim = embeddings @ train_embeddings_ref.T

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

    def apply(self, Z, train_embeddings_pos):
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = train_embeddings_pos[self.ssps_indices[ssps_mask]].clone()
        return Z


class SSPS_KMeansSampling(_SSPS_BaseSampling):

    def __init__(self, config):
        super().__init__(config)

    def init(self, device, dataset_size, batch_size):
        self.device = device

        self.kmeans = KMeans(
            dataset_size=dataset_size,
            nb_prototypes=self.config.kmeans_nb_prototypes,
            nb_iters=self.config.kmeans_nb_iters,
            verbose=self.verbose,
            batch_size=batch_size // get_world_size(),
        )

    def _run_kmeans(self, train_indices, train_embeddings):
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

    def prepare(self, train_indices_ref, train_embeddings_ref):
        self._run_kmeans(train_indices_ref, train_embeddings_ref)
        self._create_cluster_to_nearby_clusters()

    def sample(
        self,
        indices,
        embeddings,
        train_indices_ref,
        train_embeddings_ref,
        train_indices_pos,
        train_embeddings_pos,
    ):
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

    def apply(self, Z, train_embeddings_pos):
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = self.centroids[self.ssps_indices[ssps_mask]].clone()
        return Z


class SSPS_KMeansReprSampling(SSPS_KMeansSampling):

    def __init__(self, config):
        super().__init__(config)

    def _create_cluster_to_nearby_samples(self):
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

    def prepare(self, train_indices_ref, train_embeddings_ref):
        super().prepare(train_indices_ref, train_embeddings_ref)

        # self._create_cluster_to_nearby_samples()

    def sample(
        self,
        indices,
        embeddings,
        train_indices_ref,
        train_embeddings_ref,
        train_indices_pos,
        train_embeddings_pos,
    ):
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

    def apply(self, Z, train_embeddings_pos):
        ssps_mask = self.ssps_indices != -1
        Z[ssps_mask] = train_embeddings_pos[self.ssps_indices[ssps_mask]].clone()
        return Z
