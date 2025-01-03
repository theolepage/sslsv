import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import torch
import torch.profiler as profiler

from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_dataloader, load_model, evaluate


def train_with_profiling(
    args: argparse.Namespace,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    path: str = "./profiling",
):
    """
    Train a model with profiling enabled from the CLI.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.
        wait (int): Wait time in seconds before starting profiling. Defaults to 1.
        warmup (int): Warmup time in seconds for profiling. Defaults to 1.
        active (int): Active time in seconds for profiling. Defaults to 3.
        repeat (int): Number of profiling repeats. Defaults to 1.
        path (str): Path to save the profiling results. Defaults to './profiling'.

    Returns:
        None
    """
    config = load_config(args.config)
    config.trainer.epochs = 1

    train_dataloader = load_dataloader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)
    model = torch.nn.DataParallel(model)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        evaluate=evaluate,
        device=device,
    )

    with profiler.profile(
        schedule=profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range((wait + warmup + active) * repeat):
            trainer.start(resume=False)
            prof.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    args = parser.parse_args()

    train_with_profiling(args)
