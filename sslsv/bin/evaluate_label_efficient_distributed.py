import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass

import argparse

import pandas as pd

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from sslsv.Config import Config
from sslsv.datasets.Sampler import SamplerConfig
from sslsv.methods.Supervised.Supervised import Supervised, SupervisedConfig
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate
from sslsv.trainer.Trainer import OptimizerEnum, Trainer


@dataclass
class SupervisedWithPretrainedEncoderConfig(SupervisedConfig):

    pass


class SupervisedWithPretrainedEncoder(Supervised):
    """
    Supervised method with a pre-trained encoder.
    """

    def __init__(self, config: Config, encoder: nn.Module):
        """
        Initialize a Supervised method with a pre-trained encoder.

        Args:
            config (Config): Method configuration.
            model (torch.nn.Module): Pre-trained encoder module.

        Returns:
            None
        """
        super().__init__(SupervisedWithPretrainedEncoderConfig(), lambda: encoder)


def train(args: argparse.Namespace, label_percentage: float):
    """
    Train a model with a limited amount of samples per speaker (label-efficient evaluation).

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.
        label_percentage (float): Number of samples per speaker.

    Returns:
        None
    """
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Load encoder
    if args.encoder_config:
        encoder_config = load_config(args.encoder_config, verbose=False)
        encoder_model = load_model(encoder_config)
        checkpoint = torch.load(
            encoder_config.model_ckpt_path / f"model_avg.pt", map_location="cpu"
        )
        encoder_model.load_state_dict(checkpoint["model"])
        if args.freeze:
            for p in encoder_model.parameters():
                p.requires_grad = False

    # Load config
    config = load_config(args.config)

    # Reduce LR if fine-tuning pre-trained encoder
    if args.encoder_config:
        config.trainer.learning_rate /= 10

    # Determine number of training samples
    df = pd.read_csv(config.dataset.base_path / config.dataset.train)
    if "Set" in df.columns:
        df = df[df["Set"] == "train"]
    nb_utterances = int(len(df["File"]) * label_percentage)

    print("label_percentage:", label_percentage)
    print("nb_utterances:", nb_utterances)

    # Load train dataloader
    config.dataset.sampler = SamplerConfig(nb_utterances=nb_utterances)
    train_dataloader = load_train_dataloader(config)

    # Load model
    model = load_model(config)
    if args.encoder_config:
        config.encoder = encoder_config
        model.encoder = encoder_model.encoder
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    # Determine model name / path
    model_name = (
        config.model_name if not args.encoder_config else encoder_config.model_name
    )
    model_path = (
        config.model_path if not args.encoder_config else encoder_config.model_path
    )
    model_name_suffix = (
        f"_label-efficient-{label_percentage}-frozen"
        if args.freeze
        else f"_label-efficient-{label_percentage}"
    )
    config.model_name = model_name + model_name_suffix
    config.model_path = model_path.parent / (model_path.name + model_name_suffix)
    config.model_ckpt_path = config.model_path / "checkpoints"
    config.model_path.mkdir(parents=True, exist_ok=True)

    # Start training
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
    parser.add_argument(
        "--encoder_config", type=str, help="Path to encoder config file."
    )
    parser.add_argument(
        "--label_percentages",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        nargs="*",
        help="Percentage of labels provided for each training.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze the encoder during training.",
    )
    args = parser.parse_args()

    for label_percentage in args.label_percentages:
        train(args, float(label_percentage))


# 0.01    1h
# 0.02    2h
# 0.05    5h
# 0.1    10h
# 0.2    20h
# 0.5    50h
# 1.0   100h
# total 188h = 8 days
