import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from ptflops import get_model_complexity_info, flops_to_string, params_to_string

from pathlib import Path
import argparse

from sslsv.utils.helpers import load_config, load_model, load_train_dataloader


class Trainer:

    def __init__(self, config, train_dataloader):
        self.config = config
        self.device = None
        self.train_dataloader = train_dataloader


class Model(nn.Module):

    def __init__(self, indices, labels, model):
        super().__init__()

        self.indices = indices
        self.labels = labels
        self.module = model

    def forward(self, X):
        Z = self.module(X, training=True)
        loss = self.module.train_step(
            Z=Z,
            step=0,
            step_rel=0,
            indices=self.indices,
            labels=self.labels,
        )


def compute_flops(args: argparse.Namespace):
    """
    Compute FLOPs, MACs and # params of a model.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    config = load_config(args.config, verbose=False)

    train_dataloader = load_train_dataloader(config)

    indices, X, info = next(iter(train_dataloader))

    model = Model(indices, info["labels"], load_model(config))
    model.module.trainer = Trainer(config, train_dataloader)
    model.module.on_train_start()

    macs, params = get_model_complexity_info(
        model,
        (1,),
        input_constructor=lambda input_shape: X,
        as_strings=False,
        backend='aten',
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"FLOPs: {flops_to_string(macs * 2, units='GMac').replace('Mac', 'FLOPs')}")
    print(f"MACs: {flops_to_string(macs, units='MMac').replace('Mac', 'MACs')}")
    print(f"Params: {params_to_string(params, units='M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    args = parser.parse_args()

    compute_flops(args)
