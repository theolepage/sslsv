from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

import random
import numpy as np

from tqdm import tqdm

from sslsv.methods._BaseMethod import BaseMethod
from sslsv.datasets.Dataset import Dataset, DatasetConfig
from sslsv.utils.distributed import get_world_size, is_dist_initialized, is_main_process


def seed_dataloader_worker(worker_id: int):
    """
    Set the seed for a PyTorch DataLoader worker for reproducibility.

    Args:
        worker_id (int): ID of the DataLoader worker.

    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class EvaluationTaskConfig:
    """
    Base configuration foe evaluation tasks.

    Attributes:
        __type__ (str): Type of evaluation task.
        batch_size (int): Batch size for evaluation.
        num_frames (int): Number of frames to extract from each audio file.
        frame_length (Optional[int]): Length of the frames to extract from the audio files.
    """

    __type__: str = None

    batch_size: int = 64
    num_frames: int = 1
    frame_length: Optional[int] = 64000


@dataclass
class BaseEvaluationConfig:
    """
    Base configuration for evaluations.

    Attributes:
        validation (List[EvaluationTaskConfig]): List of evaluation task configurations for validation.
        test (List[EvaluationTaskConfig]): List of evaluation task configurations for test.
    """

    validation: List[EvaluationTaskConfig] = field(default_factory=lambda: [])

    test: List[EvaluationTaskConfig] = field(default_factory=lambda: [])


class BaseEvaluation:
    """
    Base class for evaluations.

    Attributes:
        model (BaseMethod): Model to evaluate.
        config (BaseEvaluationConfig): Evaluation configuration.
        task_config (EvaluationTaskConfig): Evaluation task configuration.
        device (torch.device): Device on which tensors will be allocated.
        verbose (bool): Whether to print verbose output.
        validation (bool): Whether the evaluation is for validation or test.
    """

    def __init__(
        self,
        model: BaseMethod,
        config: BaseEvaluationConfig,
        task_config: EvaluationTaskConfig,
        device: torch.device,
        verbose: bool = False,
        validation: bool = False,
    ):
        """
        Initialize a base evaluation.

        Args:
            model (BaseMethod): Model to evaluate.
            config (BaseEvaluationConfig): Evaluation configuration.
            task_config (EvaluationTaskConfig): Evaluation task configuration.
            device (torch.device): Device on which tensors will be allocated.
            verbose (bool): Whether to print verbose output. Defaults to False.
            validation (bool): Whether the evaluation is for validation or test Defaults to False.

        Returns:
            None
        """
        self.model = model
        self.config = config
        self.task_config = task_config
        self.device = device
        self.verbose = verbose
        self.validation = validation

        if self.task_config.frame_length is None:
            self.task_config.batch_size = 1
            self.task_config.num_frames = 1

    def _ddp_sync_embeddings(
        self,
        embeddings: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Sync embeddings across all distributed processes.

        Args:
            embeddings (Dict[str, torch.Tensor]): Dictionary containing embeddings of the local process.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing synchronized embeddings across all processes.
        """
        embeddings_all = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(embeddings_all, embeddings)
        embeddings = {}
        for d in embeddings_all:
            embeddings.update(d)
        return embeddings

    def _extract_embeddings_pre(self, X: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing applied before model inference.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return X

    def _extract_embeddings_inference(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform model inference.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(X)

    def _extract_embeddings_post(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Postprocessing applied after model inference.

        Args:
            Y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return Y

    def _extract_embeddings(
        self,
        files: List[str],
        labels: Optional[List[int]] = None,
        desc: Optional[str] = None,
        numpy: bool = False,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Extract embeddings for evaluation.

        Args:
            files (List[str]): List of file paths.
            labels (Optional[List[int]]): List of labels. Defaults to None.
            desc (Optional[str]): Description for the progress bar during extraction. Defaults to None.
            numpy (bool): Whether to return embeddings as numpy arrays. Defaults to False.

        Returns:
            Dict[str, Union[torch.Tensor, np.ndarray]]: Dictionary mapping file paths to embeddings.

        Raises:
            Exception: If batch size is not set to 1 when frame length is None.
        """
        dataset_config = DatasetConfig(
            frame_length=self.task_config.frame_length,
            base_path=self.config.dataset.base_path,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
        )

        dataset = Dataset(
            dataset_config,
            files,
            labels,
            num_frames=self.task_config.num_frames,
        )

        sampler = None
        if is_dist_initialized():
            sampler = DistributedSampler(dataset, shuffle=False)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=max(1, self.task_config.batch_size // get_world_size()),
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
