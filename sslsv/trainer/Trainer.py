from typing import Any, Callable, Dict
from dataclasses import dataclass
from enum import Enum

from datetime import datetime
from pathlib import Path
import json
import subprocess
import os

os.environ["WANDB_SILENT"] = "true"
import wandb

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast

from sslsv.methods._BaseMethod import BaseMethod
from sslsv.utils.distributed import (
    is_main_process,
    is_dist_initialized,
    get_world_size,
)

from .EpochLogger import EpochLogger


class TrackedModeEnum(Enum):
    """
    Enumeration representing tracked mode options for early stopping.

    Attributes:
        MIN (str): Model has improved if the metric is lower.
        MAX (str): Model has improved if the metric is higher.
    """

    MIN = "min"
    MAX = "max"


class OptimizerEnum(Enum):
    """
    Enumeration representing different training optimizers.

    Attributes:
        ADAM (str): Adam optimizer.
        SGD (str): SGD optimizer.
    """

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class TrainerConfig:
    """
    Trainer configuration.

    Attributes:
        epochs (int): Number of epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        optimizer (OptimizerEnum): Optimizer type.
        patience (int): Number of epochs to wait for improvement before early stopping.
        tracked_metric (str): Metric used for tracking improvement for early stopping.
        tracked_mode (TrackedModeEnum): Mode for tracking the the improvement of the tracked metric.
        ddp_sync_batchnorm (bool): Whether to synchronize BatchNorm statistics when using DDP.
        mixed_precision (bool): Whether to use mixed precision training.
        init_weights (str): Path to initial weights for the model.
        wandb_id (str): ID for the WandB run.
        wandb_project (str): Project name for WandB run.
    """

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0
    optimizer: OptimizerEnum = OptimizerEnum.ADAM

    patience: int = 100
    tracked_metric: str = "val/sv_cosine/voxceleb1_test_O/eer"
    tracked_mode: TrackedModeEnum = TrackedModeEnum.MIN

    ddp_sync_batchnorm: bool = True
    mixed_precision: bool = False
    init_weights: str = None

    wandb_id: str = None
    wandb_project: str = "sslsv"


class Trainer:
    """
    Trainer class.

    Attributes:
        model (BaseMethod): Model instance used for training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        config (Any): Gloabal configuration.
        evaluate (Callable[..., Dict[str, float]]): Function for evaluating the model.
        optimizer (torch.optim.Optimizer): Optimizer instance used for training.
        scaler (torch.cuda.amp.GradScaler): GradScaler instance for mixed precision training.
        device (torch.device): Device on which tensors will be allocated.
        best_metric (float): Best metric value achieved during training.
        nb_epochs_remaining (int): Number of epochs remaining before early stopping.
        epoch (int): Current epoch number.
        epoch_start_time (datetime): Start time of the current epoch.
        tensorboard_writer (SummaryWriter): SummaryWriter instance for Tensorboard logs.
        wandb_url (str): URL for the current WandB run.
    """

    def __init__(
        self,
        model: BaseMethod,
        train_dataloader: torch.utils.data.DataLoader,
        config: Any,  # FIXME: use Config
        evaluate: Callable[..., Dict[str, float]],
        device: torch.device,
    ):
        """
        Initialize a Trainer object.

        Args:
            model (BaseMethod): Model used for training.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            config (Any): Global configuration.
            evaluate (Callable[..., Dict[str, float]]): Function for evaluating the model.
            device (torch.device): Device on which tensors will be allocated.

        Returns:
            None
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.evaluate = evaluate
        self.device = device

        self.best_metric = (
            float("-inf")
            if self.config.trainer.tracked_mode == TrackedModeEnum.MAX
            else float("inf")
        )

    def _train_step_loop(self, logger: EpochLogger):
        """
        Perform a training step.

        Args:
            logger (EpochLogger): EpochLogger object for logging metrics.

        Returns:
            None
        """
        nb_steps = self.config.trainer.epochs * len(self.train_dataloader)

        for step_rel, (indices, X, info) in enumerate(
            logger.log(self.train_dataloader)
        ):
            step = self.epoch * len(self.train_dataloader) + step_rel

            self.model.module.on_train_step_start(step, nb_steps)

            lr, wd = self.model.module.update_optim(
                self.optimizer,
                self.config.trainer.learning_rate,
                self.config.trainer.weight_decay,
                step=step,
                nb_steps=nb_steps,
                nb_steps_per_epoch=len(self.train_dataloader),
            )

            X = X.to(self.device, non_blocking=True)
            labels = info["labels"].to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)

            # Forward and compute loss
            with autocast(enabled=(self.scaler is not None)):
                Z = self.model(X, training=True)
                loss = self.model.module.train_step(
                    Z,
                    step=step,
                    step_rel=step_rel,
                    indices=indices,
                    labels=labels,
                )

            self.optimizer.zero_grad(set_to_none=True)

            # Backward
            self.model.module.on_before_backward()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            self.model.module.on_after_backward()

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.model.module.on_train_step_end(step, nb_steps)

            logger.update(
                {
                    **self.model.module.step_metrics,
                    "train/lr": lr,
                    "train/wd": wd,
                }
            )

    def _update_training_stats_file(self, metrics: Dict[str, float]):
        """
        Update `training.json` file.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics.

        Returns:
            None
        """
        log_file_path = self.config.model_path / "training.json"
        log_file_data = {}
        if log_file_path.exists():
            with open(log_file_path, "r") as f:
                log_file_data = json.load(f)
        log_file_data[self.epoch] = metrics
        with open(log_file_path, "w") as f:
            json.dump(log_file_data, f, indent=4)

    def _log_end_epoch(self, metrics: Dict[str, float]):
        """
        Log time, epoch duration and metrics at the end of an epoch.

        Args:
            metrics (Dict[str, float]): Dictionary of metric.

        Returns:
            None
        """
        time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"Time: {time}")

        epoch_duration = datetime.now() - self.epoch_start_time
        duration = str(epoch_duration).split(".")[0]
        print(f"Duration: {duration}")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {round(metric_value, 6)}")
            self.tensorboard_writer.add_scalar(
                metric_name,
                metric_value,
                self.epoch,
            )
        wandb.log(metrics, step=self.epoch)

        self._update_training_stats_file(metrics)

    def _early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check for early stopping and save best best model in the event of an improvement.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics.

        Returns:
            bool: True if early stopping criteria is met, False if training should continue.
        """
        improved = False

        if self.config.trainer.tracked_metric in metrics.keys():
            metric = metrics[self.config.trainer.tracked_metric]
            mode = self.config.trainer.tracked_mode
            if mode == TrackedModeEnum.MAX and metric > self.best_metric:
                improved = True
            if mode == TrackedModeEnum.MIN and metric < self.best_metric:
                improved = True

        if improved:
            print(
                f"\n=> {self.config.trainer.tracked_metric}"
                f" improved from {self.best_metric}"
                f" to {metric}"
            )
            self.best_metric = metric
            self.nb_epochs_remaining = 0
            self._save_checkpoint("best")
        else:
            print(
                f"\n=> {self.config.trainer.tracked_metric}"
                f" did not improve from {self.best_metric}"
            )
            self.nb_epochs_remaining += 1

        if self.nb_epochs_remaining >= self.config.trainer.patience:
            return True

        return False

    def _train_epoch_loop(self, start_epoch: int = 0):
        """
        Main training loop that performs training epochs.

        Args:
            start_epoch (int): Epoch to start training from. Defaults to 0.

        Returns:
            None
        """
        self.nb_epochs_remaining = 0

        max_epochs = self.config.trainer.epochs

        for epoch in range(start_epoch, max_epochs):
            logger = EpochLogger()

            self.epoch = epoch
            self.epoch_start_time = datetime.now()

            if callable(getattr(self.train_dataloader.sampler, "set_epoch", None)):
                self.train_dataloader.sampler.set_epoch(epoch)

            if is_main_process():
                print(f"\nEpoch {self.epoch}")

            self.model.module.on_train_epoch_start(self.epoch, max_epochs)
            self.model.train()

            self._train_step_loop(logger)

            self.model.module.on_train_epoch_end(self.epoch, max_epochs)

            logger.synchronize()

            self.model.eval()
            val_metrics = self.evaluate(
                self.model,
                self.config,
                self.device,
                validation=True,
                verbose=False,
            )

            if is_main_process():
                train_metrics = {k: v.global_avg for k, v in logger.metrics.items()}
                metrics = {**train_metrics, **val_metrics}

                self._log_end_epoch(metrics)
                early_stopping = self._early_stopping(metrics)
                self._save_checkpoint("latest")
                self._save_checkpoint(f"epoch-{self.epoch}")
                if early_stopping:
                    break

    def _load_checkpoint(self) -> Any:
        """
        Load model checkpoint for resuming or use weights of pre-trained model.

        Returns:
            Any: Loaded checkpoint, or None if no checkpoint is found.
        """
        init_weights = self.config.trainer.init_weights
        checkpoint_path = self.config.model_ckpt_path / "model_latest.pt"

        if not checkpoint_path.exists():
            if init_weights:
                checkpoint = torch.load(
                    Path(init_weights) / "model_latest.pt", map_location="cpu"
                )
                self.model.module.load_state_dict(checkpoint["model"], strict=False)

            return None

        # Resume training
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.best_metric = checkpoint["best_metric"]
        self.model.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def _save_checkpoint(self, suffix: str):
        """
        Save a checkpoint.

        Args:
            suffix (str): String suffix to differentiate different saved checkpoints.

        Returns:
            None
        """
        Path(self.config.model_ckpt_path).mkdir(exist_ok=True, parents=True)

        torch.save(
            {
                "epoch": self.epoch + 1,
                "best_metric": self.best_metric,
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.config.model_ckpt_path / f"model_{suffix}.pt",
        )

    def _log_start_training(self, resuming: bool):
        """
        Log information at the beginning of the training.

        Args:
            resuming (bool): Indicates whether training is resuming from a checkpoint.

        Returns:
            None
        """
        # gitHash = subprocess.run(
        #     ["git", "rev-parse", "--short", "HEAD"],
        #     text=True,
        #     capture_output=True,
        #     check=True,
        # ).stdout.strip()

        if self.device == torch.device("cpu"):
            training_mode = "CPU"
        else:
            num_gpus = (
                get_world_size() if is_dist_initialized() else torch.cuda.device_count()
            )
            training_mode = "DDP" if is_dist_initialized() else "DP"
            training_mode += f" ({num_gpus} GPUs)"

        wandb_online = self.wandb_url and len(self.wandb_url) > 1

        sep_length = (
            len(f"Model: {self.config.model_name}")
            if not wandb_online
            else len(f"W&B URL: {self.wandb_url}")
        )

        print()
        print("=" * 3, "Trainer", "=" * (sep_length - 12))
        print(f"Model: {self.config.model_name}")
        # print(f"Commit: {gitHash}")
        print(f"Mode: {training_mode}")
        print(f"Iterations: {len(self.train_dataloader)}")
        print(f'Resuming: {"yes" if resuming else "no"}')
        if wandb_online:
            print(f"W&B URL: {self.wandb_url}")
        print("=" * sep_length)

    def _init_optimizer(self):
        """
        Initialize the optimizer.

        Returns:
            None
        """
        if self.config.trainer.optimizer == OptimizerEnum.SGD:
            self.optimizer = SGD(
                self.model.module.get_learnable_params(),
                momentum=0.9,
                lr=0,
                weight_decay=self.config.trainer.weight_decay,
            )
        elif self.config.trainer.optimizer == OptimizerEnum.ADAM:
            self.optimizer = Adam(
                self.model.module.get_learnable_params(),
                lr=0,
                weight_decay=self.config.trainer.weight_decay,
            )

        self.scaler = GradScaler() if self.config.trainer.mixed_precision else None

    def _init_tensorboard(self):
        """
        Initialize Tensorboard logger.

        Returns:
            None
        """
        self.tensorboard_writer = SummaryWriter(
            log_dir=str(self.config.model_path / "tensorboard")
        )

    def _init_wandb(self):
        """
        Initialize Weights and Biases (WandB).

        Returns:
            None
        """
        wandb_name = self.config.model_name.replace("/", "_")
        wandb_id = (
            self.config.trainer.wandb_id if self.config.trainer.wandb_id else wandb_name
        )

        self.wandb_url = wandb.init(
            project=self.config.trainer.wandb_project,
            id=wandb_id,
            resume="allow",
            dir=str(self.config.model_path),
            name=wandb_name,
            config=vars(self.config),
        ).get_url()

    def start(self, resume: bool = True):
        """
        Start the training.

        Args:
            resume (bool): Whether to resume training from a checkpoint. Defaults to True.

        Returns:
            None
        """
        if is_dist_initialized() and self.config.trainer.ddp_sync_batchnorm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self._init_optimizer()

        if is_main_process():
            self._init_tensorboard()
            self._init_wandb()

        self.model.module.trainer = self

        self.model.module.on_train_start()

        checkpoint = self._load_checkpoint() if resume else None

        if is_main_process():
            self._log_start_training(resuming=checkpoint is not None)

        self._train_epoch_loop(start_epoch=(checkpoint["epoch"] if checkpoint else 0))

        self.model.module.on_train_end()

        if is_main_process():
            wandb.finish()
