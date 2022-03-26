import argparse

import torch

from sslsv.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_dataloader, load_model


def train(args):
    config, checkpoint_dir = load_config(args.config)
    train_dataloader = load_dataloader(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config).to(device)
    model = torch.nn.DataParallel(model)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args)
