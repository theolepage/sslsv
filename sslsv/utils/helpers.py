from pathlib import Path
import random
import os
from dotenv import load_dotenv
import numpy as np

import ruamel.yaml
from dacite import from_dict
import prettyprinter as pp

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from sslsv.configs import Config
from sslsv.data.AudioDataset import AudioDataset
from sslsv.data.SiameseAudioDataset import SiameseAudioDataset
from sslsv.data.SupervisedSampler import SupervisedSampler
from sslsv.utils.distributed import is_main_process

from sslsv.encoders.ThinResNet34 import ThinResNet34, ThinResNet34Config
from sslsv.encoders.SimpleAudioCNN import SimpleAudioCNN, SimpleAudioCNNConfig

from sslsv.models.CPC import CPC, CPCConfig
from sslsv.models.LIM import LIM, LIMConfig
from sslsv.models.SimCLR import SimCLR, SimCLRConfig
from sslsv.models.VICReg import VICReg, VICRegConfig
from sslsv.models.VIbCReg import VIbCReg, VIbCRegConfig
from sslsv.models.BarlowTwins import BarlowTwins, BarlowTwinsConfig
from sslsv.models.MultiLosses import MultiLosses, MultiLossesConfig


REGISTERED_ENCODERS = {
    'thinresnet34':   (ThinResNet34,   ThinResNet34Config),
    'simpleaudiocnn': (SimpleAudioCNN, SimpleAudioCNNConfig),
}


REGISTERED_MODELS = {
    'cpc':         (CPC,         CPCConfig),
    'lim':         (LIM,         LIMConfig),
    'simclr':      (SimCLR,      SimCLRConfig),
    'vicreg':      (VICReg,      VICRegConfig),
    'vibcreg':     (VIbCReg,     VIbCRegConfig),
    'barlowtwins': (BarlowTwins, BarlowTwinsConfig),
    'multilosses': (MultiLosses, MultiLossesConfig),
}


def get_sub_config(data, key, registered_dict):
    type_ = data[key]['type']
    if type_ not in registered_dict.keys():
        raise (
            Exception('{} `{}` not supported'.format(key.capitalize(), type_))
        )

    res = from_dict(registered_dict[type_][1], data[key])
    res.__type__ = data[key]['type']
    return res


def load_config(path):
    load_dotenv()
    
    data = ruamel.yaml.safe_load(open(path, 'r'))
    config = from_dict(Config, data)
    config.encoder = get_sub_config(data, 'encoder', REGISTERED_ENCODERS)
    config.model = get_sub_config(data, 'model', REGISTERED_MODELS)
    config.name = Path(path).stem

    # Reproducibility / performance
    torch.backends.cudnn.benchmark = not config.reproducibility
    torch.backends.cudnn.deterministic = config.reproducibility
    torch.use_deterministic_algorithms(config.reproducibility)

    # Set seed
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create checkpoint dir
    checkpoint_dir = './checkpoints/' + config.name
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print config
    if is_main_process():
        pp.install_extras(include=['dataclasses'])
        pp.pprint(config)

    return config, checkpoint_dir


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_dataloader(config, nb_labels_per_spk=None):
    train_dataset = (
        SiameseAudioDataset(config.data)
        if config.data.siamese
        else AudioDataset(config.data)
    )
    shuffle = True
    batch_size = config.training.batch_size
    sampler = None

    if dist.is_available() and dist.is_initialized():
        shuffle = False
        batch_size = config.training.batch_size // dist.get_world_size()
        sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)
    elif nb_labels_per_spk:
        shuffle = False
        sampler = SupervisedSampler(train_dataset, batch_size, nb_labels_per_spk)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        drop_last=True,
        pin_memory=config.data.pin_memory,
        worker_init_fn=seed_dataloader_worker
    )

    return train_dataloader


def load_model(config):
    encoder_cls = REGISTERED_ENCODERS[config.encoder.__type__][0]
    create_encoder_fn = lambda: encoder_cls(config.encoder)

    model = REGISTERED_MODELS[config.model.__type__][0](
        config.model,
        create_encoder_fn
    )
    return model