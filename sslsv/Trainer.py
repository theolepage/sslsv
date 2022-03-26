from datetime import datetime
from pathlib import Path

import os
os.environ["WANDB_SILENT"] = "true"
import wandb

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

from sslsv.utils.evaluate import extract_embeddings, evaluate
from sslsv.utils.distributed import is_main_process, is_dist_initialized


class Trainer:

    def __init__(
        self,
        model,
        train_dataloader,
        config,
        checkpoint_dir,
        device
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.best_metric = (
            float('-inf')
            if self.config.training.tracked_mode == 'max'
            else float('inf')
        )

    def train_step_loop(self):
        train_metrics = {}
        self.last_progress = 0

        for i, (x, y) in enumerate(self.train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            x_1 = x[:, 0, :]
            x_2 = x[:, 1, :]

            with autocast(enabled=(self.scaler is not None)):
                z_1 = self.model(x_1, training=True)
                z_2 = self.model(x_2, training=True)
                loss, metrics = self.model.module.compute_loss(z_1, z_2)

            # Update metrics (average for epoch)
            if not train_metrics:
                train_metrics = {name:0 for name in metrics.keys()}
            for metric_name, metric_value in metrics.items():
                train_metrics[metric_name] += metric_value

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if is_main_process(): self.print_progress_bar(i)

        if is_main_process(): print()

        for name, value in train_metrics.items():
            train_metrics[name] = value / len(self.train_dataloader)

        return train_metrics

    def print_progress_bar(self, i, size=100):
        progress = (i + 1) * size // len(self.train_dataloader)
        if progress > self.last_progress:
            for _ in range(progress - self.last_progress):
                print('.', end='', flush=True)
            self.last_progress = progress

    def log_metrics(self, metrics):
        time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        print(f'Time: {time}')

        epoch_duration = datetime.now() - self.epoch_start_time
        duration = str(epoch_duration).split('.')[0]
        print(f'Duration: {duration}')

        metrics = {
            **metrics,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }

        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value}') 
            self.writer.add_scalar(metric_name, metric_value, self.epoch)

        wandb.log({**metrics, 'epoch': self.epoch})

    def track_improvement(self, metrics):
        metric = metrics[self.config.training.tracked_metric]
        mode = self.config.training.tracked_mode

        improved = False
        if mode == 'max' and metric > self.best_metric: improved = True
        if mode == 'min' and metric < self.best_metric: improved = True

        if improved:
            print(
                f'\n=> {self.config.training.tracked_metric}'
                f' improved from {self.best_metric}'
                f' to {metric}'
            )
            self.best_metric = metric
            self.nb_epochs_remaining = 0
            self.save_checkpoint()
        else:
            print(
                f'\n=> {self.config.training.tracked_metric}'
                f' did not improved from {self.best_metric}'
            )
            self.nb_epochs_remaining += 1

        if self.nb_epochs_remaining >= self.config.training.patience:
            return False

        return True

    def start_epoch(self, epoch):
        self.epoch = epoch
        self.epoch_start_time = datetime.now()
        if callable(getattr(self.train_dataloader.sampler, 'set_epoch', None)):
            self.train_dataloader.sampler.set_epoch(epoch)

    def train_epoch_loop(self, first_epoch=0):
        self.nb_epochs_remaining = 0

        for epoch in range(first_epoch, self.config.training.epochs):
            self.start_epoch(epoch)
        
            if is_main_process(): print(f'\nEpoch {self.epoch}')

            self.model.train()
            train_metrics = self.train_step_loop()

            self.model.eval()
            test_embeddings = extract_embeddings(self.model, self.config.data)
            test_metrics = evaluate(test_embeddings, self.config.data.trials)

            metrics = {**train_metrics, **test_metrics}

            if is_main_process(): self.log_metrics(metrics)
            
            self.lr_scheduler.step()
            
            if is_main_process():
                if not self.track_improvement(metrics):
                    break


    def setup(self):
        if self.config.training.optimizer == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                momentum=0.9,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_reg
            )
        elif self.config.training.optimizer == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_reg
            )
        else:
            raise Exception(
                f'Optimizer {self.config.training.optimizer} not supported'
            )
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.95)
        self.scaler = (
            GradScaler() if self.config.training.mixed_precision else None
        )

        if not is_main_process(): return

        # Init tensorboard and wandb
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir + '/logs')
        self.wandb_url = wandb.init(
            project='sslsv',
            dir=self.checkpoint_dir,
            name=self.config.name,
            config=vars(self.config)
        ).get_url()

    def load_checkpoint(self):
        checkpoint_path = Path(self.checkpoint_dir) / 'model.pt'
        checkpoint = None
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.best_metric = checkpoint['best_metric']
            self.model.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return checkpoint

    def save_checkpoint(self):
        torch.save(
            {
                'epoch': self.epoch + 1,
                'best_metric': self.best_metric,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            },
            self.checkpoint_dir + '/model.pt'
        )

    def start(self):
        self.setup()
        checkpoint = self.load_checkpoint()

        if is_main_process():
            print()
            print('=' * 10, 'Training', '=' * 10)
            print(f'Distributed: {"yes" if is_dist_initialized() else "no"}')
            print(f'Device: {self.device}')
            print(f'Number of batches: {len(self.train_dataloader)}')
            print(f'Resuming: {"no" if checkpoint is None else "yes"}')
            print(f'Checkpoint: {self.checkpoint_dir}')
            if os.getenv('WANDB_MODE') != 'offline':
                print(f'WandB url: {self.wandb_url}')

        first_epoch = 0 if checkpoint is None else checkpoint['epoch']
        self.train_epoch_loop(first_epoch)

        wandb.finish()
