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

from sslsv.utils.distributed import (
    is_main_process,
    is_dist_initialized,
    get_world_size,
)

from .EpochLogger import EpochLogger


class TrackedModeEnum(Enum):

    MIN = 'min'
    MAX = 'max'


class OptimizerEnum(Enum):

    ADAM = 'adam'
    SGD  = 'sgd'


@dataclass
class TrainerConfig:

    epochs        : int = 300
    batch_size    : int = 256
    learning_rate : float = 0.001
    weight_decay  : float = 0
    optimizer     : OptimizerEnum = OptimizerEnum.ADAM

    patience       : int = 300
    tracked_metric : str = 'val/sv_cosine/voxceleb1_test_O/eer'
    tracked_mode   : TrackedModeEnum = TrackedModeEnum.MIN

    ddp_sync_batchnorm : bool = True
    mixed_precision    : bool = False
    init_weights       : str = None

    wandb_id      : str = None
    wandb_project : str = 'sslsv'


class Trainer:

    def __init__(
        self,
        model,
        train_dataloader,
        config,
        evaluate,
        device
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.evaluate = evaluate
        self.device = device

        self.best_metric = (
            float('-inf')
            if self.config.trainer.tracked_mode == TrackedModeEnum.MAX
            else float('inf')
        )

    def _train_step_loop(self, logger):
        nb_steps = self.config.trainer.epochs * len(self.train_dataloader)

        for i, (idx, X, info) in enumerate(logger.log(self.train_dataloader)):
            step = self.epoch * len(self.train_dataloader) + i

            self.model.module.on_train_step_start(step, nb_steps)

            lr = self.model.module.update_optim(
                self.optimizer,
                self.config.trainer,
                step=step,
                nb_steps=nb_steps,
                nb_steps_per_epoch=len(self.train_dataloader)
            )

            X = X.to(self.device)
            labels = info['labels'].to(self.device)

            # Forward and compute loss
            with autocast(enabled=(self.scaler is not None)):
                Z = self.model(X, training=True)
                loss, metrics = self.model.module.train_step(
                    Z,
                    labels=labels,
                    step=i,
                    samples=idx
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

            logger.update({**metrics, 'train/lr': lr})

    def _update_training_stats_file(self, metrics):
        log_file_path = self.config.experiment_path / 'training.json'
        log_file_data = {}
        if log_file_path.exists():
            with open(log_file_path, 'r') as f:
                log_file_data = json.load(f)
        log_file_data[self.epoch] = metrics
        with open(log_file_path, 'w') as f:
            json.dump(log_file_data, f, indent=4)

    def _log_end_epoch(self, metrics):
        time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        print(f'Time: {time}')

        epoch_duration = datetime.now() - self.epoch_start_time
        duration = str(epoch_duration).split('.')[0]
        print(f'Duration: {duration}')

        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {round(metric_value, 6)}') 
            self.tensorboard_writer.add_scalar(
                metric_name,
                metric_value,
                self.epoch
            )
        wandb.log(metrics, step=self.epoch)

        self._update_training_stats_file(metrics)

    def _early_stopping(self, metrics):
        improved = False

        if self.config.trainer.tracked_metric in metrics.keys():
            metric = metrics[self.config.trainer.tracked_metric]
            mode = self.config.trainer.tracked_mode
            if mode == TrackedModeEnum.MAX and metric > self.best_metric: improved = True
            if mode == TrackedModeEnum.MIN and metric < self.best_metric: improved = True

        if improved:
            print(
                f'\n=> {self.config.trainer.tracked_metric}'
                f' improved from {self.best_metric}'
                f' to {metric}'
            )
            self.best_metric = metric
            self.nb_epochs_remaining = 0
            self._save_checkpoint('best')
        else:
            print(
                f'\n=> {self.config.trainer.tracked_metric}'
                f' did not improve from {self.best_metric}'
            )
            self.nb_epochs_remaining += 1

        if self.nb_epochs_remaining >= self.config.trainer.patience:
            return False

        return True

    def _train_epoch_loop(self, start_epoch=0):
        self.nb_epochs_remaining = 0

        max_epochs = self.config.trainer.epochs

        for epoch in range(start_epoch, max_epochs):
            logger = EpochLogger()

            self.epoch = epoch
            self.epoch_start_time = datetime.now()

            if callable(getattr(self.train_dataloader.sampler, 'set_epoch', None)):
                self.train_dataloader.sampler.set_epoch(epoch)

            if is_main_process(): print(f'\nEpoch {self.epoch}')
            
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
                verbose=False
            )

            if is_main_process():
                train_metrics = {k:v.global_avg for k, v in logger.metrics.items()}
                metrics = {**train_metrics, **val_metrics}

                self._log_end_epoch(metrics)
                self._save_checkpoint('latest')
                if not self._early_stopping(metrics): break

    def _load_checkpoint(self):
        init_weights = self.config.trainer.init_weights
        checkpoint_path = self.config.experiment_path / 'model_latest.pt'

        if not checkpoint_path.exists():
            if init_weights:
                checkpoint = torch.load(
                    Path(init_weights) / 'model_latest.pt',
                    map_location='cpu'
                )
                self.model.module.load_state_dict(checkpoint['model'], strict=False)

            return None

        # Resume training
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.best_metric = checkpoint['best_metric']
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint

    def _save_checkpoint(self, suffix):
        torch.save(
            {
                'epoch': self.epoch + 1,
                'best_metric': self.best_metric,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            self.config.experiment_path / f'model_{suffix}.pt'
        )

    def _log_start_training(self, resuming):
        gitHash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True,
            capture_output=True,
            check=True
        ).stdout.strip()

        wandb_online = self.wandb_url and len(self.wandb_url) > 1
        
        sep_length = (
            len(f'Experiment: {self.config.experiment_name}')
            if not wandb_online
            else len(f'W&B URL: {self.wandb_url}')
        )

        print()
        print('=' * 3, 'Trainer', '=' * (sep_length - 12))
        print(f'Experiment: {self.config.experiment_name}')
        print(f'Commit: {gitHash}')
        print(f'Mode: {f"DDP ({get_world_size()} GPUs)" if is_dist_initialized() else "DP"}')
        print(f'Iterations: {len(self.train_dataloader)}')
        print(f'Resuming: {"yes" if resuming else "no"}')
        if wandb_online:
            print(f'W&B URL: {self.wandb_url}')
        print('=' * sep_length)

    def _init_optimizer(self):
        if self.config.trainer.optimizer == OptimizerEnum.SGD:
            self.optimizer = SGD(
                self.model.module.get_learnable_params(),
                momentum=0.9,
                lr=0,
                weight_decay=self.config.trainer.weight_decay
            )
        elif self.config.trainer.optimizer == OptimizerEnum.ADAM:
            self.optimizer = Adam(
                self.model.module.get_learnable_params(),
                lr=0,
                weight_decay=self.config.trainer.weight_decay
            )
        
        self.scaler = (
            GradScaler() if self.config.trainer.mixed_precision else None
        )

    def _init_tensorboard(self):
        self.tensorboard_writer = SummaryWriter(
            log_dir=str(self.config.experiment_path / 'tensorboard')
        )

    def _init_wandb(self):
        wandb_name = self.config.experiment_name.replace('/', '_')
        wandb_id = self.config.trainer.wandb_id if self.config.trainer.wandb_id else wandb_name
        
        self.wandb_url = wandb.init(
            project=self.config.trainer.wandb_project,
            id=wandb_id,
            resume='allow',
            dir=str(self.config.experiment_path),
            name=wandb_name,
            config=vars(self.config)
        ).get_url()

    def start(self, resume=True):
        if is_dist_initialized() and self.config.trainer.ddp_sync_batchnorm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self._init_optimizer()

        if is_main_process():
            self._init_tensorboard()
            self._init_wandb()

        self.model.module.on_train_start(self)
        
        checkpoint = self._load_checkpoint() if resume else None

        if is_main_process():
            self._log_start_training(resuming=checkpoint is not None)

        self._train_epoch_loop(
            start_epoch=(checkpoint['epoch'] if checkpoint else 0)
        )

        self.model.module.on_train_end(self)

        if is_main_process():
            wandb.finish()