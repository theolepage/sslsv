from datetime import datetime
from datetime import timedelta
from pathlib import Path
import json

import os
os.environ["WANDB_SILENT"] = "true"
import wandb

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast

from sslsv.utils.helpers import evaluate
from sslsv.utils.distributed import (
    is_main_process,
    is_dist_initialized,
    get_rank,
    get_world_size
)






import time
from collections import defaultdict
import torch.distributed as dist
class MetricValue(object):

    def __init__(self, fmt='{global_avg:.6f}'):
        self.fmt = fmt

        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.count += 1
        self.total += value

    def synchronize(self):
        if not is_dist_initialized(): return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(global_avg=self.global_avg)

class MetricLogger:

    def __init__(self, delimiter=' '):
        self.metrics = defaultdict(MetricValue)
        self.delimiter = delimiter

    def update(self, metrics):
        for k, v in metrics.items():
            assert isinstance(v, (torch.Tensor, float, int))

            if isinstance(v, torch.Tensor): v = v.item()
            self.metrics[k].update(v)

    def __str__(self):
        res = []
        for name, metric in self.metrics.items():
            res.append('{}: {}'.format(name, metric))
        return self.delimiter.join(res)

    def __getitem__(self, key):
        return self.metrics[key]

    def synchronize(self):
        for metric in self.metrics.values():
            metric.synchronize()

    def log(self, iterable, print_freq=100):
        i = 0

        last_iter_end_time = time.time()
        iter_time = MetricValue()
        
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{metrics}'
        ])

        for obj in iterable:
            yield obj
            iter_time.update(time.time() - last_iter_end_time)
            if get_rank() == 0 and (i % print_freq == 0 or i == len(iterable) - 1):
                eta = int(iter_time.global_avg * (len(iterable) - i))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=str(timedelta(seconds=eta)),
                        metrics=str(self)
                    )
                )

            i += 1
            last_iter_end_time = time.time()








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

        nb_steps = self.config.training.epochs * len(self.train_dataloader)

        # metric_logger = MetricLogger()
        # for i, (idx, X, info) in enumerate(metric_logger.log(self.train_dataloader)):
        for i, (idx, X, info) in enumerate(self.train_dataloader):
            step = self.epoch * len(self.train_dataloader) + i

            self.model.module.on_train_step_start(step, nb_steps)

            lr = self.model.module.update_optim(
                self.optimizer,
                self.config.training,
                step=step,
                nb_steps=nb_steps,
                nb_steps_per_epoch=len(self.train_dataloader)
            )

            X = X.to(self.device)
            labels = info['labels'].to(self.device)

            # Forward and compute loss
            with autocast(enabled=(self.scaler is not None)):
                Z = self.model(
                    X,
                    training=True
                )
                loss, metrics = self.model.module.train_step(
                    Z,
                    labels=labels,
                    step=i,
                    samples=idx
                )

            # Update metrics (average for epoch)
            if not train_metrics:
                train_metrics = {name:0 for name in metrics.keys()}
            for metric_name, metric_value in metrics.items():
                train_metrics[metric_name] += metric_value

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

            # metric_logger.update({
            #     'loss': loss
            # })

            if is_main_process():
                print(
                    f"\rStep {i}/{len(self.train_dataloader)} - "
                    f"Loss: {round(loss.item(), 4)} - ",
                    f"LR: {round(lr, 4)}",
                    end='',
                    flush=True
                )
                # self.print_progress_bar(i)

        # metric_logger.synchronize()

        if is_main_process(): print()

        for name, value in train_metrics.items():
            if isinstance(value, torch.Tensor): value = value.item()
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

        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value}') 
            self.writer.add_scalar(metric_name, metric_value, self.epoch)

        log_file_path = Path(self.checkpoint_dir) / 'training.json'
        log_file_data = {}
        if log_file_path.exists():
            with open(log_file_path, 'r') as f:
                log_file_data = json.load(f)
        log_file_data[self.epoch] = metrics
        with open(log_file_path, 'w') as f:
            json.dump(log_file_data, f, indent=4)
        
        wandb.log(metrics, step=self.epoch)

    def track_improvement(self, metrics):
        improved = False

        if self.config.training.tracked_metric in metrics.keys():
            metric = metrics[self.config.training.tracked_metric]
            mode = self.config.training.tracked_mode
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
            self.save_checkpoint('best')
        else:
            print(
                f'\n=> {self.config.training.tracked_metric}'
                f' did not improve from {self.best_metric}'
            )
            self.nb_epochs_remaining += 1
        self.save_checkpoint('latest')

        if self.nb_epochs_remaining >= self.config.training.patience:
            return False

        return True

    def gather_metrics(self, metrics):
        for k, v in metrics.items():
            metric = torch.tensor([v]).to(get_rank())
            if is_main_process():
                metric_all = [metric.clone() for _ in range(get_world_size())]
                torch.distributed.gather(metric, metric_all, dst=0)
                metrics[k] = torch.mean(torch.cat(metric_all)).item()
            else:
                torch.distributed.gather(metric, [], dst=0)
        return metrics

    def train_epoch_loop(self, first_epoch=0):
        self.nb_epochs_remaining = 0

        max_epochs = self.config.training.epochs

        for epoch in range(first_epoch, max_epochs):
            self.epoch = epoch
            self.epoch_start_time = datetime.now()
            if callable(getattr(self.train_dataloader.sampler, 'set_epoch', None)):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.module.on_train_epoch_start(self.epoch, max_epochs)
        
            if is_main_process(): print(f'\nEpoch {self.epoch}')

            self.model.train()
            train_metrics = self.train_step_loop()

            self.model.module.on_train_epoch_end(self.epoch, max_epochs)

            if is_dist_initialized():
                train_metrics = self.gather_metrics(train_metrics)

            if is_main_process():
                self.model.eval()
                val_metrics = evaluate(
                    self.model,
                    self.config,
                    self.device,
                    validation=True,
                    verbose=False
                )

                metrics = {**train_metrics, **val_metrics}

                self.log_metrics(metrics)
                if not self.track_improvement(metrics): break

    def setup(self):
        params = self.model.module.get_learnable_params()

        if self.config.training.optimizer == 'sgd':
            self.optimizer = SGD(
                params,
                momentum=0.9,
                lr=0,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adam':
            self.optimizer = Adam(
                params,
                lr=0,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise Exception(
                f'Optimizer {self.config.training.optimizer} not supported'
            )
        self.scaler = (
            GradScaler() if self.config.training.mixed_precision else None
        )

        if not is_main_process(): return

        # Init tensorboard and wandb
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir + '/logs')
        self.wandb_url = wandb.init(
            project=self.config.wandb_project,
            id=(self.config.wandb_id if self.config.wandb_id else self.config.name),
            resume='allow',
            dir=self.checkpoint_dir,
            name=self.config.name,
            config=vars(self.config)
        ).get_url()

    def load_checkpoint(self):
        init_weights = self.config.training.init_weights
        checkpoint_path = Path(self.checkpoint_dir) / 'model_latest.pt'

        if not checkpoint_path.exists():
            if init_weights:
                checkpoint_path = (
                    Path(get_checkpoint_dir(init_weights)) /
                    init_weights_path /
                    'model_latest.pt'
                )

                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.module.load_state_dict(checkpoint['model'])

            return None

        # Resume training
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.best_metric = checkpoint['best_metric']
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint

    def save_checkpoint(self, suffix):
        torch.save(
            {
                'epoch': self.epoch + 1,
                'best_metric': self.best_metric,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            self.checkpoint_dir + '/model_' + suffix + '.pt'
        )

    def start(self, resume=True):
        if is_dist_initialized() and self.config.training.ddp_sync_batchnorm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.setup()

        self.model.module.on_train_start(self)
        
        checkpoint = None
        if resume: checkpoint = self.load_checkpoint()

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

        self.model.module.on_train_end(self)

        if is_main_process():
            wandb.finish()