import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass

import argparse

import torch
import torch.nn as nn

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


def get_model_name_suffix(
    nb_samples_per_spk: int,
    fine_tune: bool,
    supervised: bool,
) -> str:
    """
    Return a suffix for a model name based on the training configuration.

    Args:
        nb_samples_per_spk (int): Number of samples per speaker.
        fine_tune (bool): Whether the model is fine-tuned.
        supervised (bool): Whether the model is supervised.

    Returns:
        str: Suffix for the model name.
    """
    suffix = "_label-efficient-"
    suffix += str(nb_samples_per_spk) + "-"
    if supervised:
        suffix += "supervised"
    else:
        suffix += "finetuned" if fine_tune else "frozen"
    return suffix


def train(
    args: argparse.Namespace,
    nb_samples_per_spk: int,
    fine_tune: bool = False,
    supervised: bool = False,
):
    """
    Train a model with a limited amount of samples per speaker (label-efficient evaluation).

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.
        nb_samples_per_spk (int): Number of samples per speaker.
        fine_tune (bool): Whether the model is fine-tuned. Defaults to False.
        supervised (bool): Whether the model is supervised. Defaults to False.

    Returns:
        None
    """
    config = load_config(args.config)

    config.dataset.sampler = SamplerConfig(nb_samples_per_spk=nb_samples_per_spk)
    config.dataset.augmentation.enable = False
    config.dataset.ssl = False
    config.trainer.optimizer = OptimizerEnum.ADAM
    config.trainer.epochs = args.epochs
    config.trainer.batch_size = args.batch_size
    config.trainer.patience = args.patience
    config.trainer.learning_rate = args.lr
    if fine_tune:
        config.trainer.learning_rate /= 10

    train_dataloader = load_train_dataloader(config)

    # Load model (to use as an encoder)
    model = load_model(config)
    if not supervised:
        checkpoint = torch.load(config.experiment_path / "model_latest.pt")
        model.load_state_dict(checkpoint["model"])
        for p in model.parameters():
            p.requires_grad = fine_tune

    # Create classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = SupervisedWithPretrainedEncoder(config, model.encoder).to(device)
    classifier = torch.nn.DataParallel(classifier)

    model_name_suffix = get_model_name_suffix(nb_samples_per_spk, fine_tune, supervised)
    config.experiment_name = config.experiment_name + model_name_suffix
    config.experiment_path = config.experiment_path.parent / (
        config.experiment_path.name + model_name_suffix
    )
    config.experiment_path.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=classifier,
        train_dataloader=train_dataloader,
        config=config,
        evaluate=evaluate,
        device=device,
    )
    trainer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs for trainings.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate used during trainings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used during trainings.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs without a lower EER before ending training.",
    )
    parser.add_argument(
        "--nb_labels",
        default=[100, 50, 20, 10, 8, 6, 4, 2],
        nargs="*",
        help="Numbers of labels provided per speaker for each training.",
    )
    args = parser.parse_args()

    for nb_labels in args.nb_labels:
        train(args, int(nb_labels), supervised=True)
        train(args, int(nb_labels), fine_tune=False)
        train(args, int(nb_labels), fine_tune=True)
