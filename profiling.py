import argparse

import torch
import torch.profiler as profiler

from sslsv.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_dataloader, load_model


def train_with_profiling(
    args,
    wait=1,
    warmup=1,
    active=3,
    repeat=1,
    path='./checkpoints/profiling'
):
    config = load_config(args.config)
    config.training.epochs = 1

    train_dataloader = load_dataloader(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config).to(device)
    model = torch.nn.DataParallel(model)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        device=device
    )

    with profiler.profile(
        schedule=profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range((wait + warmup + active) * repeat):
            trainer.start(resume=False)
            prof.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train_with_profiling(args)
