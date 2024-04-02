from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np

from tqdm import tqdm

from sslsv.data.AudioDataset import AudioDataset


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class EvaluationTaskConfig:

    __type__: str = None


class BaseEvaluation:

    def __init__(
        self,
        model,
        config,
        task_config,
        device='cpu',
        verbose=False,
        validation=False
    ):
        self.model = model
        self.config = config
        self.task_config = task_config
        self.device = device
        self.verbose = verbose
        self.validation = validation

    def _extract_embeddings(self, files, labels=None, desc=None, numpy=False):
        dataset = AudioDataset(
            base_path=self.config.data.base_path,
            files=files,
            labels=labels,
            frame_length=self.config.evaluation.frame_length,
            num_frames=self.config.evaluation.num_frames
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.evaluation.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            worker_init_fn=seed_dataloader_worker
        )

        embeddings = {}

        dataloader = tqdm(dataloader, desc=desc) if self.verbose else dataloader
        for idx, X, info in dataloader:
            B, N, L = X.size()

            X = X.to(self.device)
            X = X.reshape((B * N, L))

            with torch.no_grad():
                if self.task_config.__type__ in ['sv_cosine', 'sv_plda']:
                    Y = self.model(X)
                else: # self.task_config.__type__ == 'classification'
                    Y = self.model(X, training=True)
                    Y = F.softmax(Y, dim=-1)
                    Y = torch.argmax(Y, dim=-1)

            Y = Y.reshape((B, N, -1))

            if self.config.evaluation.mean_of_features:
                Y = Y.mean(dim=1, keepdim=True)

            if self.task_config.__type__ in ['sv_cosine', 'sv_plda']:
                Y = F.normalize(Y, p=2, dim=-1)

            embeddings.update({
                info['files'][i]:(Y[i].cpu().numpy() if numpy else Y[i].cpu())
                for i in range(B)
            })

        return embeddings