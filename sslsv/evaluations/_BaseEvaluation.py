from dataclasses import dataclass, field
from typing import List, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

import random
import numpy as np

from tqdm import tqdm

from sslsv.datasets.Dataset import Dataset, DatasetConfig
from sslsv.utils.distributed import get_world_size, is_dist_initialized, is_main_process


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class EvaluationTaskConfig:

    __type__: str = None


@dataclass
class BaseEvaluationConfig:

    batch_size: int = 64
    num_frames: int = 10
    frame_length: Union[int, None] = 64000

    validation: List[EvaluationTaskConfig] = field(default_factory=lambda: [])

    test: List[EvaluationTaskConfig] = field(default_factory=lambda: [])


class BaseEvaluation:

    def __init__(
        self,
        model,
        config,
        task_config,
        device="cpu",
        verbose=False,
        validation=False,
    ):
        self.model = model
        self.config = config
        self.task_config = task_config
        self.device = device
        self.verbose = verbose
        self.validation = validation

    def _ddp_sync_embeddings(self, embeddings):
        embeddings_all = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(embeddings_all, embeddings)
        embeddings = {}
        for d in embeddings_all:
            embeddings.update(d)
        return embeddings

    def _extract_embeddings_pre(self, X):
        return X

    def _extract_embeddings_inference(self, X):
        return self.model(X)

    def _extract_embeddings_post(self, Y):
        return Y

    def _extract_embeddings(self, files, labels=None, desc=None, numpy=False):
        dataset_config = DatasetConfig(
            frame_length=self.config.evaluation.frame_length,
            base_path=self.config.dataset.base_path,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
        )

        dataset = Dataset(
            dataset_config,
            files,
            labels,
            num_frames=self.config.evaluation.num_frames,
        )

        sampler = None
        if is_dist_initialized():
            sampler = DistributedSampler(dataset, shuffle=False)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=max(1, self.config.evaluation.batch_size // get_world_size()),
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            worker_init_fn=seed_dataloader_worker,
        )

        embeddings = {}

        if is_main_process():
            dataloader = tqdm(dataloader, desc=desc) if self.verbose else dataloader
        for idx, X, info in dataloader:
            if X.ndim == 2:
                X = X.unsqueeze(1)

            X = X.to(self.device)
            B, N, L = X.size()

            X = self._extract_embeddings_pre(X)

            X = X.reshape((B * N, L))

            with torch.no_grad():
                Y = self._extract_embeddings_inference(X)

            Y = Y.reshape((B, N, -1))

            Y = self._extract_embeddings_post(Y)

            embeddings.update(
                {
                    info["files"][i]: (Y[i].cpu().numpy() if numpy else Y[i].cpu())
                    for i in range(B)
                }
            )

        if is_dist_initialized():
            embeddings = self._ddp_sync_embeddings(embeddings)

        return embeddings
