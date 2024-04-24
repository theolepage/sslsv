import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import torch
from torch import nn

from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate


class ModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.module = model

    def forward(self, x, training=False):
        return self.module(x, training)


def train(args: argparse.Namespace):
    config = load_config(args.config)
    train_dataloader = load_train_dataloader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)

    if device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)
    else:
        model = ModelWrapper(model)

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
