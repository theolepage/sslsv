import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from glob import glob

import torch

from sslsv.utils.helpers import load_config


def average_model(args: argparse.Namespace):
    """
    Average a model checkpoints.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    config = load_config(args.config, verbose=not args.silent)

    checkpoints = glob(str(config.model_ckpt_path / "model_epoch-*.pt"))
    checkpoints = sorted(
        checkpoints, key=lambda p: int(p.split("/")[-1].split("_epoch-")[-1][:-3])
    )

    if args.limit_nb_epochs:
        checkpoints = checkpoints[: args.limit_nb_epochs]

    checkpoints = checkpoints[-args.count :]

    average_model = None
    for ckpt_path in checkpoints:
        current_model = torch.load(ckpt_path, map_location="cpu")["model"]
        if average_model is None:
            average_model = current_model
        else:
            for k in average_model.keys():
                average_model[k] += current_model[k]

    for k in average_model.keys():
        if average_model[k] is not None:
            average_model[k] = torch.true_divide(average_model[k], len(checkpoints))

    torch.save(
        {"model": average_model},
        config.model_path / "checkpoints" / "model_avg.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of epochs to average.",
    )
    parser.add_argument(
        "--limit_nb_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs used for model averaging.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    args = parser.parse_args()

    average_model(args)
