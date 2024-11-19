import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any

import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from sslsv.utils.helpers import load_config, load_train_dataloader, load_model

from sslsv.methods._BaseMethod import BaseMethod
from sslsv.utils.distributed import is_main_process

from create_ssps_buffers_distributed import Trainer

import idr_torch


def train(args: argparse.Namespace):
    """
    Train a model from the CLI (using DistributedDataParallel).

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    world_size = idr_torch.size
    rank = idr_torch.rank

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    config = load_config(args.config, verbose=not args.silent)
    train_dataloader = load_train_dataloader(config)

    if Path(config.trainer.last_checkpoint).exists():
        return

    model = load_model(config).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        device=torch.device("cuda", rank),
        verbose=not args.silent,
    )
    trainer.start()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    args = parser.parse_args()

    train(args)
