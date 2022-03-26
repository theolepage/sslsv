import argparse
import os

import torch
from torch.nn.parallel import DistributedDataParallel

from sslsv.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_dataloader, load_model


def train(args):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["LOCAL_RANK"])

    torch.distributed.init_process_group(
        'nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.device(rank)

    config, checkpoint_dir = load_config(args.config)
    train_dataloader = load_dataloader(config)

    model = load_model(config).to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=rank
    )
    trainer.start()

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args)