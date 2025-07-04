import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel

from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate


def train(args: argparse.Namespace):
    """
    Train a model from the CLI (using DistributedDataParallel).

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    config = load_config(args.config)
    train_dataloader = load_train_dataloader(config)

    model = load_model(config).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        evaluate=evaluate,
        device=torch.device("cuda", rank),
    )
    trainer.start()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    args = parser.parse_args()

    train(args)
