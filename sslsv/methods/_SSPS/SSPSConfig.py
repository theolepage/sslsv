from dataclasses import dataclass
from enum import Enum


class SSPSSamplingMethodEnum(Enum):
    """
    Enumeration representing SSPS sampling methods.

    Options:
        KNN (str): Nearest-Neighbors sampling.
        KMEANS (str): K-Means Clustering sampling (centroid).
        KMEANS_REPR (str): K-Means clustering sampling (reresentation).
    """

    KNN = "knn"
    KMEANS = "kmeans"
    KMEANS_REPR = "kmeans-repr"


@dataclass
class SSPSConfig:
    """
    Self-Supervised Positive Sampling (SSPS) configuration.

    Attributes:
        start_epoch (int): Training epoch at which SSPS will be enabled.
        sampling (SSPSSamplingMethodEnum): Sampling method.
        pos_queue_size (int): Number of elements in the positive queue (Q').
        kmeans_nb_prototypes (int): Number of K-Means prototypes (K).
        kmeans_nb_iters (int): Number of K-Means iterations.
        inter_sampling_size (int): Sampling window between clusters (SSPS-Clustering: M).
        inter_sampling_prob_fn (str): Probability function for inter sampling (uniform or exp).
        inter_sampling_prob_exp_lambda (float): Exponential lambda for inter sampling.
        intra_sampling_size (int): Sampling window within clusters (SSPS-NN: M).
        intra_sampling_prob_fn (str): Probability function for intra sampling (uniform or exp).
        intra_sampling_prob_exp_lambda (float): Exponential lambda for intra sampling.
        verbose (bool): Whether to log status messages and progress bars.
    """

    start_epoch: int = 100

    sampling: SSPSSamplingMethodEnum = SSPSSamplingMethodEnum.KMEANS_REPR

    pos_queue_size: int = 50000

    kmeans_nb_prototypes: int = 50000
    kmeans_nb_iters: int = 10

    inter_sampling_size: int = 0
    inter_sampling_prob_fn: str = "uniform"
    inter_sampling_prob_exp_lambda: float = 0.7

    intra_sampling_size: int = 10000000
    intra_sampling_prob_fn: str = "uniform"
    intra_sampling_prob_exp_lambda: float = 0.7

    verbose: bool = False
