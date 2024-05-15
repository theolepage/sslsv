import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import torch
from torch import nn
from torch import Tensor as T

from sslsv.methods._BaseMethod import BaseMethod
from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate


class MethodWrapper(nn.Module):
    """
    Class representing a wrapper for a PyTorch model to be accessed from a module attribute
    for compatiblity with DataParallel and DistributedDataParallel wrappers.

    Attributes:
        module (BaseMethod): Method to wrap.
    """

    def __init__(self, method: BaseMethod):
        """
        Initialize the class with a method.

        Args:
            method (BaseMethod): Method object to be assigned to the 'module' attribute.
        """
        super().__init__()

        self.module = method

    def forward(self, x: T, training: bool = False):
        """
        Compute the forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): Whether the method should be in training mode or not.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.module(x, training)


def train(args: argparse.Namespace):
    """
    Train a model from the CLI.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    config = load_config(args.config)
    train_dataloader = load_train_dataloader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)

    if device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)
    else:
        model = MethodWrapper(model)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        evaluate=evaluate,
        device=device,
    )
    trainer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    args = parser.parse_args()

    train(args)
