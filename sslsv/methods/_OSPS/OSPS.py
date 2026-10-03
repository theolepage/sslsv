from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sklearn.metrics import normalized_mutual_info_score

from sslsv.utils.distributed import gather


class OSPSMetrics:
    """
    Metrics to determine the quality of the pseudo-positives sampled by OSPS.

    Speaker and video accuracies are determined for each sample of the batch, as in SSPS,
    and are thus directly comparable to `ssps_speaker_acc` and `ssps_video_acc`.

    Attributes:
        nb_clusters (int): Number of clusters (K).
        speakers (T): Speaker label of each training sample. Shape: (N).
        videos (T): Video label of each training sample. Shape: (N).
        assignments (T): Cluster assigned to each training sample. Shape: (N).
    """

    def __init__(self, nb_clusters: int):
        """
        Initialize a OSPSMetrics object.

        Args:
            nb_clusters (int): Number of clusters (K).

        Returns:
            None
        """
        self.nb_clusters = nb_clusters

        self.speakers = None
        self.videos = None
        self.assignments = None

    def init(self, dataset_size: int, train_csv: str, device: torch.device):
        """
        Initialize the assignments buffer and the speaker and video labels.

        Args:
            dataset_size (int): Number of samples in the train set.
            train_csv (str): Path to the train set csv file.
            device (torch.device): Device on which tensors will be allocated.

        Returns:
            None
        """
        df_train = pd.read_csv(train_csv)

        speakers = pd.factorize(df_train["Speaker"])[0]
        videos = pd.factorize(
            np.array([file.split("/")[-2] for file in df_train["File"]])
        )[0]

        self.speakers = torch.from_numpy(speakers).long().to(device)
        self.videos = torch.from_numpy(videos).long().to(device)

        self.assignments = -torch.ones(dataset_size, dtype=torch.long, device=device)

    def record(self, indices: T, assignments: T):
        """
        Record the clusters assigned to the samples of the current batch.

        Args:
            indices (T): Indices of current batch.
            assignments (T): Clusters assigned to the samples of current batch.

        Returns:
            None
        """
        self.assignments[indices] = assignments

    def accuracies(self, indices: T, indices_pos: T) -> Tuple[float, float]:
        """
        Determine the speaker and video accuracies of the pseudo-positives.

        Args:
            indices (T): Indices of the anchors.
            indices_pos (T): Indices of their pseudo-positives.

        Returns:
            Tuple[float, float]: Speaker and video accuracies.
        """
        if len(indices) == 0:
            return 0.0, 0.0

        speaker_acc = (self.speakers[indices] == self.speakers[indices_pos]).float()
        video_acc = (self.videos[indices] == self.videos[indices_pos]).float()

        return speaker_acc.mean().item(), video_acc.mean().item()

    def compute(self) -> Dict[str, float]:
        """
        Determine clustering metrics from the assignments recorded during the last epoch
        and discard them to start a new epoch.

        Returns:
            Dict[str, float]: Clustering metrics.
        """
        # A copy is required as .cpu() is a no-op when tensors are already on CPU and
        # assignments are discarded below
        assignments = self.assignments.cpu().numpy().copy()
        self.assignments.fill_(-1)

        mask = assignments >= 0
        if mask.sum() == 0:
            return {}

        assignments = assignments[mask]
        speakers = self.speakers.cpu().numpy()[mask]
        videos = self.videos.cpu().numpy()[mask]

        return {
            "osps_nmi_speaker": normalized_mutual_info_score(speakers, assignments),
            "osps_nmi_video": normalized_mutual_info_score(videos, assignments),
            "osps_clusters_used": float(
                len(np.unique(assignments)) / self.nb_clusters
            ),
        }

@dataclass
class OSPSConfig:
    """
    Online Self-Supervised Positive Sampling (OSPS) configuration.

    Attributes:
        start_epoch (int): Training epoch at which OSPS will be enabled.
        nb_clusters (int): Number of clusters of the codebook (K).
        memory_size (int): Number of embeddings retained for each cluster (R), among
            which the pseudo-positive is sampled. Equivalent to the intra sampling pool
            of SSPS (`ssps_intra_sampling_pool`).
        centroids_momentum (float): Momentum of the centroids of the codebook.
        inter_sampling_size (int): Number of nearby clusters to sample from (M), among
            which the cluster of the pseudo-positive is selected. Equivalent to
            `inter_sampling_size` of SSPS. 0 samples from the cluster of the anchor
            itself, which requires the clusters to be at the granularity of speakers.
        coverage (float): Fraction of the samples of a batch for which the positive is
            substituted.
        confidence_gating (bool): Whether the samples whose positive is substituted are
            the ones whose cluster is the closest to the one of the anchor, instead of a
            random fraction of the batch. The budget is spent where the guess is the most
            likely to hold the same speaker, so the same coverage yields purer positives.
            It favours the closest clusters though, which are also the ones most likely to
            hold the recording of the anchor.
        reciprocal_rank (int): Rank up to which a candidate cluster must have the
            cluster of the anchor among its own nearest ones to be kept. Being a test of
            symmetry rather than of distance, it discards the candidates that are unlikely
            to hold the same speaker without favouring the closest clusters. 1 keeps only
            the pairs of clusters that choose each other, None disables the test.
        pool_expansion (bool): Whether the pseudo-positive is drawn from the union of the
            memories of every candidate that was kept, instead of from a single one of
            them. Verifying the candidates then widens the pool of recordings to draw from
            instead of narrowing it.
        rank_by_neighbours (bool): Whether the candidate cluster is the one sharing the
            most neighbours with the cluster of the anchor, instead of a random one
            among those that were kept. Two clusters of a speaker are surrounded by the
            same clusters, whereas a cluster that is close by accident is not, which
            distance alone cannot tell apart. The candidate it selects is not the
            closest one, so the recording of the anchor is not favoured.
        jaccard_neighbours (int): Number of nearest clusters describing the neighbourhood
            of a cluster, used by the Jaccard filter and by the ranking.
        jaccard_threshold (float): Minimum fraction of neighbours a candidate cluster must
            share with the cluster of the anchor to be kept. None disables the filter.
        hard_mining (float): Fraction of the memory of the selected cluster, ordered
            from least to most similar to the anchor, among which the pseudo-positive is
            sampled. None disables mining and draws uniformly from the whole memory;
            values close to 0 always take the single most dissimilar embedding. Raises
            the difficulty of the positives without changing which cluster they come
            from, but the least similar embeddings of a cluster are also the ones most
            likely to belong to another speaker.
        nb_clusters_final (int): Number of clusters the codebook is contracted to, by
            merging the closest ones at the start of every epoch. None keeps the codebook
            at its initial size. Contracting it turns clusters that hold a single
            recording into clusters that hold the recordings of a speaker, so that a
            pseudo-positive drawn from the cluster of the anchor is progressively a
            different session of the same speaker.
        annealing_epochs (int): Number of epochs over which the codebook is contracted.
        repulsion_weight (float): Weight of the repulsion term, which pushes the
            representations away from the centroids of the clusters they do not belong
            to. None disables it. Unlike a classifier over pseudo-speakers it never pulls
            samples together, so a cluster that holds two speakers cannot weld them.
        repulsion_temperature (float): Temperature of the repulsion term.
        repulsion_exclude (int): Number of nearest clusters, including the one of the
            anchor, that are left out of the repulsion. They are the ones likely to hold
            the same speaker, so repelling them would fight the pseudo-positives.
        metrics (bool): Whether to determine NMI and speaker and video accuracies.
        verbose (bool): Whether to log status messages.
    """

    start_epoch: int = 75

    nb_clusters: int = 5000

    memory_size: int = 4

    centroids_momentum: float = 0.9

    inter_sampling_size: int = 1

    coverage: float = 1.0
    confidence_gating: bool = False

    reciprocal_rank: Optional[int] = None
    pool_expansion: bool = False
    rank_by_neighbours: bool = False
    jaccard_neighbours: int = 20
    jaccard_threshold: Optional[float] = None

    hard_mining: Optional[float] = None

    nb_clusters_final: Optional[int] = None
    annealing_epochs: int = 50

    repulsion_weight: Optional[float] = None
    repulsion_temperature: float = 0.1
    repulsion_exclude: int = 25

    metrics: bool = False
    verbose: bool = False


class OSPS(nn.Module):
    """
    Online Self-Supervised Positive Sampling (OSPS).

    Queue-free reformulation of SSPS: the reference queue and the epoch-boundary K-Means
    are replaced by a codebook of centroids maintained by online spherical K-Means in the
    space of the encoder, and the positive queue is replaced by a small memory of the
    embeddings most recently assigned to each cluster. The pseudo-positive of a sample is
    then an embedding of a nearby cluster, i.e. another recording of (hopefully) the same
    speaker, as in SSPS.

    The global view is used as reference representation (4s, neither augmented nor
    cropped), so no additional frame nor forward pass is required.

    Attributes:
        config (OSPSConfig): OSPS configuration.
        enabled (bool): Whether positives are currently substituted.
        centroids (T): Centroids of the codebook. Shape: (K, D).
        centroids_ready (T): Whether each centroid is initialized. Shape: (K).
        centroids_count (T): Number of samples assigned to each centroid during the
            current epoch. Shape: (K).
        memory (T): Embeddings stored for each cluster. Shape: (K, R, E).
        memory_indices (T): Indices of the stored embeddings. Shape: (K, R).
        memory_repr (T): Reference representations of the stored embeddings.
            Shape: (K, R, D).
        positive_repr (T): Reference representations of the pseudo-positives of the
            current batch. Shape: (N, D).
        sampled (T): Whether a pseudo-positive was substituted. Shape: (N).
        assignments (T): Cluster assigned to each sample of the current batch. Shape: (N).
        memory_ptr (T): Position of the next insertion in each cluster. Shape: (K).
        memory_size (T): Number of embeddings stored in each cluster. Shape: (K).
        metrics (OSPSMetrics): OSPS metrics.
        step_metrics (Dict[str, float]): Metrics related to OSPS.
    """

    def __init__(self, config: OSPSConfig, encoder_dim: int, embeddings_dim: int):
        """
        Initialize a OSPS module.

        Args:
            config (OSPSConfig): OSPS configuration.
            encoder_dim (int): Dimension of the representations of the encoder.
            embeddings_dim (int): Dimension of the embeddings of the head.

        Returns:
            None
        """
        super().__init__()

        self.config = config
        self.embeddings_dim = embeddings_dim

        self.enabled = False

        K = config.nb_clusters

        # The codebook is the state that takes a long time to converge, so it is saved
        # with the model and survives the restarts a long training requires. The memory
        # below refills within a few hundred steps and is left out of the checkpoints,
        # which it would otherwise dominate in size.
        self.register_buffer("centroids", torch.zeros(K, encoder_dim))
        self.register_buffer("centroids_ready", torch.zeros(K, dtype=torch.bool))
        self.register_buffer("centroids_count", torch.zeros(K, dtype=torch.long))
        self.register_buffer("centroids_alive", torch.ones(K, dtype=torch.bool))

        self.neighbours = None

        self.memory = None
        self.memory_indices = None
        self.memory_repr = None

        self.positive_repr = None
        self.sampled = None
        self.assignments = None
        self.memory_ptr = None
        self.memory_size = None

        self.metrics = None

        self.global_metrics = {}
        self.step_metrics = {}

    def initialize(
        self,
        dataset_size: int,
        train_csv: str,
        device: torch.device,
    ):
        """
        Initialize the memory and the metrics.

        Args:
            dataset_size (int): Number of samples in the train set.
            train_csv (str): Path to the train set csv file.
            device (torch.device): Device on which tensors will be allocated.

        Returns:
            None
        """
        K, R = self.config.nb_clusters, self.config.memory_size
        encoder_dim = self.centroids.size(-1)

        self.memory = torch.zeros(K, R, self.embeddings_dim, device=device)
        self.memory_indices = -torch.ones(K, R, dtype=torch.long, device=device)

        # Reference representations of the stored embeddings, used to mine hard
        # pseudo-positives and to determine how hard the sampled ones are
        self.memory_repr = (
            torch.zeros(K, R, encoder_dim, device=device)
            if self.config.metrics or self.config.hard_mining is not None
            else None
        )
        self.memory_ptr = torch.zeros(K, dtype=torch.long, device=device)
        self.memory_size = torch.zeros(K, dtype=torch.long, device=device)

        if self.config.metrics:
            self.metrics = OSPSMetrics(K)
            self.metrics.init(dataset_size, train_csv, device)

    def set_epoch(self, epoch: int):
        """
        Enable OSPS and recycle the clusters that were not assigned any sample during the
        last epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            None
        """
        self.enabled = epoch >= self.config.start_epoch

        if self.centroids_count is None or self.centroids_count.sum() == 0:
            return

        dead = (self.centroids_count == 0) & self.centroids_alive
        self.centroids_ready &= ~dead
        self.memory_size[dead] = 0
        self.centroids_count.zero_()

        if self.config.verbose:
            print(f"OSPS: recycled {dead.sum().item()} clusters")

        self._contract_codebook(epoch)
        self._update_neighbours()

    @torch.no_grad()
    def _contract_codebook(self, epoch: int):
        """
        Merge the closest clusters until the codebook reaches the size scheduled for the
        current epoch.

        Merging is agglomerative and greedy: every cluster proposes its nearest neighbour
        and the closest pairs are merged first, so that a cluster takes part in at most one
        merge per epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            None
        """
        if self.config.nb_clusters_final is None:
            return

        K, K_end = self.config.nb_clusters, self.config.nb_clusters_final

        progress = (epoch - self.config.start_epoch + 1) / self.config.annealing_epochs
        target = round(K + (K_end - K) * min(1.0, max(0.0, progress)))

        usable = self.centroids_alive & self.centroids_ready
        nb_merges = int(self.centroids_alive.sum().item() - target)
        if nb_merges <= 0 or usable.sum() < 2:
            return

        alive = torch.nonzero(usable).view(-1)
        C = F.normalize(self.centroids[alive], p=2, dim=-1)

        # Nearest neighbour of every cluster, in chunks to keep the similarity matrix small
        best_sim = torch.empty(len(alive), device=C.device)
        best_idx = torch.empty(len(alive), dtype=torch.long, device=C.device)
        for start in range(0, len(alive), 1024):
            stop = min(start + 1024, len(alive))
            sim = C[start:stop] @ C.T
            sim[torch.arange(stop - start), torch.arange(start, stop)] = -2
            best_sim[start:stop], best_idx[start:stop] = sim.max(dim=-1)

        order = torch.argsort(best_sim, descending=True).tolist()
        best_idx = best_idx.tolist()

        merged = torch.zeros(len(alive), dtype=torch.bool)
        nb = 0
        for a in order:
            b = best_idx[a]
            if nb >= nb_merges:
                break
            if merged[a] or merged[b]:
                continue
            merged[a] = merged[b] = True
            self._merge(alive[a].item(), alive[b].item())
            nb += 1

        if self.config.verbose:
            print(
                f"OSPS: merged {nb} clusters, "
                f"{self.centroids_alive.sum().item()} left (target {target})"
            )

    @torch.no_grad()
    def _merge(self, i: int, j: int):
        """
        Absorb the cluster j into the cluster i and retire j.

        Args:
            i (int): Cluster that is kept.
            j (int): Cluster that is retired.

        Returns:
            None
        """
        R = self.config.memory_size

        count_i = self.centroids_count[i].clamp(min=1)
        count_j = self.centroids_count[j].clamp(min=1)
        self.centroids[i] = F.normalize(
            (count_i * self.centroids[i] + count_j * self.centroids[j]).float()
            / (count_i + count_j),
            p=2,
            dim=-1,
        )
        self.centroids_count[i] += self.centroids_count[j]

        # Keep as many stored embeddings of both clusters as the memory can hold
        size_i, size_j = self.memory_size[i].item(), self.memory_size[j].item()
        take = min(R - size_i, size_j)
        if take > 0:
            self.memory[i, size_i : size_i + take] = self.memory[j, :take]
            self.memory_indices[i, size_i : size_i + take] = self.memory_indices[j, :take]
            if self.memory_repr is not None:
                self.memory_repr[i, size_i : size_i + take] = self.memory_repr[j, :take]
            self.memory_size[i] = size_i + take
            self.memory_ptr[i] = (size_i + take) % R

        self.centroids_alive[j] = False
        self.centroids_ready[j] = False
        self.memory_size[j] = 0
        self.centroids_count[j] = 0

    def compute_metrics(self):
        """
        Determine clustering metrics from the assignments recorded during the last epoch.

        Returns:
            None
        """
        if self.metrics is None:
            return

        self.global_metrics = self.metrics.compute()

        if self.config.verbose and self.global_metrics:
            print(
                "OSPS: "
                + ", ".join(f"{k}={v:.4f}" for k, v in self.global_metrics.items())
            )

    @torch.no_grad()
    def _initialize_centroids(self, Y: T):
        """
        Initialize the centroids that are not ready with representations of the batch.

        Args:
            Y (T): Reference representations of the batch. Shape: (N, D).

        Returns:
            None
        """
        missing = torch.nonzero(~self.centroids_ready & self.centroids_alive).view(-1)
        if len(missing) == 0:
            return

        nb = min(len(missing), len(Y))
        self.centroids[missing[:nb]] = Y[:nb]
        self.centroids_ready[missing[:nb]] = True

    @torch.no_grad()
    def _assign(self, Y: T) -> T:
        """
        Assign representations to their nearest centroid.

        Args:
            Y (T): Reference representations. Shape: (N, D).

        Returns:
            T: Assigned clusters. Shape: (N).
        """
        sim = Y @ self.centroids.T
        sim[:, ~self.centroids_ready] = -2
        return torch.argmax(sim, dim=-1)

    @torch.no_grad()
    def _update_neighbours(self):
        """
        Determine the nearest clusters of every cluster, used by the Jaccard filter.

        Returns:
            None
        """
        if not self.config.jaccard_threshold and not self.config.rank_by_neighbours:
            return

        usable = self.centroids_alive & self.centroids_ready
        k = min(self.config.jaccard_neighbours, int(usable.sum().item()) - 1)
        if k <= 0:
            self.neighbours = None
            return

        C = F.normalize(self.centroids, p=2, dim=-1)

        self.neighbours = torch.zeros_like(self.centroids_ready).repeat(
            len(self.centroids), 1
        )
        for start in range(0, len(C), 512):
            stop = min(start + 512, len(C))
            sim = C[start:stop] @ C.T
            sim[:, ~usable] = -2
            sim[torch.arange(stop - start), torch.arange(start, stop)] = -2
            self.neighbours[start:stop].scatter_(
                1, torch.topk(sim, k, dim=-1)[1], True
            )
        self.neighbours &= usable.unsqueeze(0)

    @torch.no_grad()
    def _reciprocal(self, candidates: T, assignments: T, usable: T) -> T:
        """
        Determine the candidates that have the cluster of the anchor among their nearest.

        Args:
            candidates (T): Candidate clusters. Shape: (N, M).
            assignments (T): Clusters assigned to the anchors. Shape: (N).
            usable (T): Clusters that can be sampled from. Shape: (K).

        Returns:
            T: Whether each candidate is reciprocal. Shape: (N, M).
        """
        if not self.config.reciprocal_rank:
            return torch.ones_like(candidates, dtype=torch.bool)

        N, M = candidates.shape

        back = self.centroids[candidates.reshape(-1)] @ self.centroids.T
        back.scatter_(1, candidates.reshape(-1, 1), -2)
        back[:, ~usable] = -2

        r = min(self.config.reciprocal_rank, back.size(-1))
        nearest = torch.topk(back, r, dim=-1)[1].view(N, M, r)

        return (nearest == assignments.view(N, 1, 1)).any(dim=-1)

    @torch.no_grad()
    def _overlapping(self, candidates: T, assignments: T) -> T:
        """
        Determine the candidates whose neighbourhood is close enough to the one of the
        cluster of the anchor.

        Two clusters of a speaker sit among the same clusters even when they are not the
        nearest to each other, whereas two clusters of different speakers that happen to be
        close rarely share a neighbourhood.

        Args:
            candidates (T): Candidate clusters. Shape: (N, M).
            assignments (T): Clusters assigned to the anchors. Shape: (N).

        Returns:
            T: Whether each candidate overlaps enough. Shape: (N, M).
        """
        if not self.config.jaccard_threshold or self.neighbours is None:
            return torch.ones_like(candidates, dtype=torch.bool)

        anchor = self.neighbours[assignments].unsqueeze(1)
        other = self.neighbours[candidates]

        intersection = (anchor & other).sum(-1)
        union = (anchor | other).sum(-1).clamp(min=1)

        return intersection / union >= self.config.jaccard_threshold

    @torch.no_grad()
    def _shared(self, candidates: T, assignments: T) -> T:
        """
        Count the neighbours each candidate shares with the cluster of the anchor.

        Args:
            candidates (T): Candidate clusters. Shape: (N, M).
            assignments (T): Clusters assigned to the anchors. Shape: (N).

        Returns:
            T: Number of shared neighbours. Shape: (N, M).
        """
        anchor = self.neighbours[assignments].unsqueeze(1)
        other = self.neighbours[candidates]

        return (anchor & other).sum(-1)

    @torch.no_grad()
    def _sample(self, assignments: T, Y: T, indices: T) -> Tuple[T, T, T]:
        """
        Sample a pseudo-positive from a nearby cluster for each sample.

        Args:
            assignments (T): Clusters assigned to the samples. Shape: (N).
            Y (T): Reference representations of the anchors. Shape: (N, D).
            indices (T): Indices of the anchors, which are never selected as their own
                pseudo-positive. Shape: (N).

        Returns:
            Tuple[T, T, T]: Sampled embeddings, their indices (-1 if not sampled) and
                their similarity to the anchor. Shapes: (N, E), (N) and (N).
        """
        N = len(assignments)
        device = assignments.device

        usable = self.centroids_ready & (self.memory_size > 0)

        if self.config.inter_sampling_size == 0:
            selected = assignments
            valid = usable[selected]
            # The cluster is the one of the anchor, so how much it can be trusted is how
            # well the anchor belongs to it
            confidence = (Y * self.centroids[selected]).sum(-1)
        else:
            sim = self.centroids[assignments] @ self.centroids.T

            # Prevent sampling from the cluster of the anchor and from clusters which are
            # not ready or empty
            sim.scatter_(1, assignments.unsqueeze(-1), -2)
            sim[:, ~usable] = -2

            M = min(self.config.inter_sampling_size, sim.size(-1))
            nearby, nearby_idx = torch.topk(sim, M, dim=-1)

            # Candidates that hold the same speaker as the anchor, judged on the structure
            # of the neighbourhood rather than on a distance, so that the clusters holding
            # another recording are not discarded along with the ones of another speaker
            keep = nearby > -2
            keep &= self._reciprocal(nearby_idx, assignments, usable)
            keep &= self._overlapping(nearby_idx, assignments)

            if self.config.rank_by_neighbours and self.neighbours is not None:
                # Take the candidate that fits the neighbourhood of the anchor best, which
                # is not necessarily the closest one
                shared = self._shared(nearby_idx, assignments)
                choice = shared.masked_fill(~keep, -1).argmax(dim=-1, keepdim=True)
                valid = keep.gather(1, choice).squeeze(-1)
            elif self.config.pool_expansion:
                # Draw from the union of the memories of every candidate that was kept, so
                # that verifying candidates widens the pool instead of narrowing it
                weights = self.memory_size[nearby_idx] * keep
                valid = weights.sum(-1) > 0
                choice = torch.multinomial(
                    torch.where(valid.unsqueeze(-1), weights, 1).float(), 1
                )
            else:
                # Draw among the candidates that were kept, or anywhere if none was
                choice = torch.multinomial(
                    torch.where(keep.any(-1, keepdim=True), keep, True).float(), 1
                )
                valid = keep.gather(1, choice).squeeze(-1)

            selected = nearby_idx.gather(1, choice).squeeze(-1)

            # How close the selected cluster is to the one of the anchor, which is how
            # likely they hold the same speaker
            confidence = nearby.gather(1, choice).squeeze(-1)

        if self.config.coverage < 1:
            if self.config.confidence_gating:
                # Substitute the positives of the anchors whose cluster is the most
                # trustworthy, rather than of a random fraction of the batch
                nb = max(1, round(self.config.coverage * N))
                kept = torch.zeros(N, dtype=torch.bool, device=device)
                kept[torch.topk(confidence.masked_fill(~valid, -2), nb)[1]] = True
                valid &= kept
            else:
                valid &= torch.rand(N, device=device) < self.config.coverage

        if valid.any():
            self.step_metrics["osps_confidence"] = confidence[valid].mean().item()

        if self.config.hard_mining is not None:
            # Least similar stored embeddings of the selected cluster, ignoring the slots
            # that are empty or hold the anchor itself
            candidates = torch.einsum("nd,nrd->nr", Y, self.memory_repr[selected])
            filled = torch.arange(self.config.memory_size, device=device).unsqueeze(0)
            filled = filled < self.memory_size[selected].unsqueeze(-1)
            filled &= self.memory_indices[selected] != indices.unsqueeze(-1)
            candidates = candidates.masked_fill(~filled, 2)

            nb = max(1, round(self.config.hard_mining * self.config.memory_size))
            hardest = torch.topk(candidates, nb, dim=-1, largest=False)[1]

            # Clusters holding fewer embeddings than nb must not select an empty slot
            sizes = filled.sum(-1).clamp(min=1, max=nb)
            slots = hardest.gather(
                1, (torch.rand(N, device=device) * sizes).long().unsqueeze(-1)
            ).squeeze(-1)
        else:
            sizes = self.memory_size[selected].clamp(min=1)
            slots = (torch.rand(N, device=device) * sizes).long()

        embeddings = self.memory[selected, slots]
        indices = torch.where(valid, self.memory_indices[selected, slots], -1)

        sims = None
        if self.memory_repr is not None:
            sims = (Y * self.memory_repr[selected, slots]).sum(-1)

        self._selected, self._slots = selected, slots

        return embeddings, indices, sims

    @torch.no_grad()
    def _update_memory(self, assignments: T, Z: T, Y: T, indices: T):
        """
        Insert the embeddings of the batch in the memory of their cluster.

        Args:
            assignments (T): Clusters assigned to the samples. Shape: (N).
            Z (T): Embeddings of the samples. Shape: (N, E).
            Y (T): Reference representations of the samples. Shape: (N, D).
            indices (T): Indices of the samples. Shape: (N).

        Returns:
            None
        """
        counts = torch.bincount(assignments, minlength=self.config.nb_clusters)


        R = self.config.memory_size

        order = torch.argsort(assignments)
        assignments = assignments[order]

        starts = torch.cumsum(counts, 0) - counts

        # Position of each sample among the samples of its cluster
        offsets = torch.arange(len(assignments), device=assignments.device)
        offsets = (offsets - starts[assignments]) % R

        slots = (self.memory_ptr[assignments] + offsets) % R

        self.memory[assignments, slots] = Z[order]
        self.memory_indices[assignments, slots] = indices[order]
        if self.memory_repr is not None:
            self.memory_repr[assignments, slots] = Y[order]

        self.memory_ptr = (self.memory_ptr + counts) % R
        self.memory_size = torch.clamp(self.memory_size + counts, max=R)

    @torch.no_grad()
    def _update_centroids(self, assignments: T, Y: T):
        """
        Update the centroids of the codebook with the representations of the batch.

        Args:
            assignments (T): Clusters assigned to the samples. Shape: (N).
            Y (T): Reference representations. Shape: (N, D).

        Returns:
            None
        """
        counts = torch.bincount(assignments, minlength=self.config.nb_clusters)

        sums = torch.zeros_like(self.centroids)
        sums.index_add_(0, assignments, Y)

        updated = counts > 0
        means = sums[updated] / counts[updated].unsqueeze(-1)

        momentum = self.config.centroids_momentum
        self.centroids[updated] = F.normalize(
            momentum * self.centroids[updated] + (1 - momentum) * means, p=2, dim=-1
        )

        self.centroids_count += counts

    @torch.no_grad()
    def substitute(self, indices: T, Z: T, Y_ref: T) -> T:
        """
        Substitute the positives of the batch with pseudo-positives.

        Args:
            indices (T): Indices of current batch.
            Z (T): Embeddings (teacher) of current batch. Shape: (N, E).
            Y_ref (T): Reference representations (Y_ref) of current batch. Shape: (N, D).

        Returns:
            T: Embeddings with substituted pseudo-positives. Shape: (N, E).
        """
        indices_all = gather(indices)
        Y_all = gather(Y_ref).float()
        Z_all = gather(Z.detach()).float()

        self._initialize_centroids(Y_all)

        assignments = self._assign(Y_ref.float())
        assignments_all = gather(assignments)

        self.assignments = assignments

        Z_pp = Z
        self.step_metrics = {**self.global_metrics}

        if self.enabled:
            embeddings, indices_pp, sims = self._sample(assignments, Y_ref.float(), indices)

            sampled = indices_pp != -1
            Z_pp = torch.where(sampled.unsqueeze(-1), embeddings.to(Z.dtype), Z)

            self.sampled = sampled
            if self.memory_repr is not None:
                self.positive_repr = self.memory_repr[self._selected, self._slots]

            self.step_metrics["osps_coverage"] = sampled.float().mean().item()

            if self.metrics is not None:
                speaker_acc, video_acc = self.metrics.accuracies(
                    indices[sampled], indices_pp[sampled]
                )
                self.step_metrics["osps_speaker_acc"] = speaker_acc
                self.step_metrics["osps_video_acc"] = video_acc
                # A pseudo-positive is only worth substituting when it holds the speaker
                # of the anchor and another of its recordings (same video implies same
                # speaker, so the difference is the fraction that is worth it)
                self.step_metrics["osps_useful_rate"] = speaker_acc - video_acc

                # How hard the pseudo-positives are, against the similarity of an anchor
                # to its own centroid as a reference scale
                if sampled.any():
                    self.step_metrics["osps_positive_sim"] = sims[sampled].mean().item()
                self.step_metrics["osps_anchor_sim"] = (
                    (Y_ref.float() * self.centroids[assignments]).sum(-1).mean().item()
                )

        if self.metrics is not None:
            self.metrics.record(indices_all, assignments_all)

        # Positives are inserted after sampling so that a sample cannot be its own
        # pseudo-positive and centroids are updated last to keep assignments consistent
        self._update_memory(assignments_all, Z_all, Y_all, indices_all)
        self._update_centroids(assignments_all, Y_all)

        return Z_pp

    def repel(self, Y: T) -> T:
        """
        Compute the repulsion term.

        Every representation is pushed away from the centroids of the clusters it does not
        belong to, leaving out the nearest ones which are likely to hold the same speaker.
        The term only separates clusters and never pulls samples toward anything, so a
        cluster that holds two speakers cannot bring them together.

        Args:
            Y (T): Representations of the local views of the students. Shape: (N * V, D).

        Returns:
            T: Loss tensor.
        """
        if self.config.repulsion_weight is None or not self.enabled:
            return Y.new_zeros(())

        assignments = self.assignments
        V = len(Y) // len(assignments)
        assignments = assignments.repeat(V)

        Y = F.normalize(Y, p=2, dim=-1)
        centroids = self.centroids.to(Y.dtype)

        sim = Y @ centroids.T

        with torch.no_grad():
            # Clusters that hold the same speaker as the anchor, which the pseudo-positives
            # are drawn from, are left out along with the ones that are not in use
            near = self.centroids[assignments].to(Y.dtype) @ centroids.T
            near[:, ~(self.centroids_alive & self.centroids_ready)] = -2

            E = min(self.config.repulsion_exclude, near.size(-1))
            excluded = torch.zeros_like(sim, dtype=torch.bool)
            excluded.scatter_(1, torch.topk(near, E, dim=-1)[1], True)
            excluded |= ~(self.centroids_alive & self.centroids_ready).unsqueeze(0)

        sim = sim.masked_fill(excluded, -torch.inf)

        loss = torch.logsumexp(sim / self.config.repulsion_temperature, dim=-1).mean()

        self.step_metrics["osps_repulsion"] = loss.item()

        return self.config.repulsion_weight * loss
