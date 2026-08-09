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


class Trainer:
    """
    Trainer class.

    Attributes:
        model (BaseMethod): Model instance used for training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        config (Any): Gloabal configuration.
        device (torch.device): Device on which tensors will be allocated.
        verbose (bool): Whether to show logging information.
    """

    def __init__(
        self,
        model: BaseMethod,
        train_dataloader: torch.utils.data.DataLoader,
        config: Any,  # FIXME: use Config
        device: torch.device,
        verbose: bool,
    ):
        """
        Initialize a Trainer object.

        Args:
            model (BaseMethod): Model used for training.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            config (Any): Global configuration.
            device (torch.device): Device on which tensors will be allocated.
            verbose (bool): Whether to show logging information.

        Returns:
            None
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.device = device
        self.verbose = verbose

    def start(self):
        """
        Start the training.

        Returns:
            None
        """
        self.model.module.trainer = self

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.module.on_train_start()
        self.model.module.ssps.enabled_next_epoch = True

        # Base checkpoint is the last checkpoint without its SSPS suffix
        # (e.g. model_epoch-74_ssps-5k.pt -> model_epoch-74.pt)
        last_checkpoint_path = Path(self.config.trainer.last_checkpoint)
        checkpoint_path = last_checkpoint_path.with_name(
            last_checkpoint_path.stem.split("_ssps-")[0] + last_checkpoint_path.suffix
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.module.load_state_dict(checkpoint["model"], strict=False)

        self.train_dataloader.sampler.set_epoch(checkpoint["epoch"])

        dataloader = (
            tqdm(self.train_dataloader, desc="Creating SSPS buffers")
            if is_main_process() and self.verbose
            else self.train_dataloader
        )
        for step_rel, (indices, X, _) in enumerate(dataloader):
            X = X.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)

            Z = self.model(X, training=True)
            self.model.module.train_step(
                Z,
                step=step_rel,
                step_rel=step_rel,
                indices=indices,
            )

        if is_main_process():
            torch.save(
                {
                    **checkpoint,
                    "model": self.model.module.state_dict(),
                },
                self.config.trainer.last_checkpoint,
            )


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

    config = load_config(args.config, verbose=not args.silent)

    if config.method.ssps is None or config.trainer.last_checkpoint is None:
        return

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
