from pathlib import Path
import random
import os
import numpy as np
import pandas as pd

from enum import Enum
import ruamel.yaml
from dacite import from_dict
from dacite import Config as DaciteConfig
import prettyprinter as pp

import torch
from torch.utils.data import DataLoader, DistributedSampler

from sslsv.Config import Config
from sslsv.utils.distributed import is_dist_initialized, is_main_process, get_world_size

# Datasets
from sslsv.datasets.Dataset import Dataset
from sslsv.datasets.SSLDataset import SSLDataset
from sslsv.datasets.Sampler import Sampler
from sslsv.datasets.DistributedSamplerWrapper import DistributedSamplerWrapper

# Encoders
from sslsv.encoders.TDNN import TDNN, TDNNConfig
from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.encoders.SimpleAudioCNN import SimpleAudioCNN, SimpleAudioCNNConfig
from sslsv.encoders.ECAPATDNN import ECAPATDNN, ECAPATDNNConfig

# Methods
from sslsv.methods.Supervised.Supervised import Supervised, SupervisedConfig
from sslsv.methods.CPC.CPC import CPC, CPCConfig
from sslsv.methods.LIM.LIM import LIM, LIMConfig
from sslsv.methods.SimCLR.SimCLR import SimCLR, SimCLRConfig
from sslsv.methods.MoCo.MoCo import MoCo, MoCoConfig
from sslsv.methods.WMSE.WMSE import WMSE, WMSEConfig
from sslsv.methods.BarlowTwins.BarlowTwins import BarlowTwins, BarlowTwinsConfig
from sslsv.methods.VICReg.VICReg import VICReg, VICRegConfig
from sslsv.methods.VIbCReg.VIbCReg import VIbCReg, VIbCRegConfig
from sslsv.methods.BYOL.BYOL import BYOL, BYOLConfig
from sslsv.methods.SimSiam.SimSiam import SimSiam, SimSiamConfig
from sslsv.methods.DINO.DINO import DINO, DINOConfig
from sslsv.methods.DeepCluster.DeepCluster import DeepCluster, DeepClusterConfig
from sslsv.methods.SwAV.SwAV import SwAV, SwAVConfig
from sslsv.methods.Combiner.Combiner import Combiner, CombinerConfig
from sslsv.methods.SimCLRCustom.SimCLRCustom import SimCLRCustom, SimCLRCustomConfig

# Evaluations
from sslsv.evaluations.CosineSVEvaluation import (
    CosineSVEvaluation,
    CosineSVEvaluationTaskConfig
)
from sslsv.evaluations.PLDASVEvaluation import (
    PLDASVEvaluation,
    PLDASVEvaluationTaskConfig
)
from sslsv.evaluations.ClassificationEvaluation import (
    ClassificationEvaluation,
    ClassificationEvaluationTaskConfig
)


REGISTERED_EVALUATIONS = {
    'sv_cosine':      (CosineSVEvaluation,       CosineSVEvaluationTaskConfig),
    'sv_plda':        (PLDASVEvaluation,         PLDASVEvaluationTaskConfig),
    'classification': (ClassificationEvaluation, ClassificationEvaluationTaskConfig),
}


REGISTERED_ENCODERS = {
    'tdnn':           (TDNN,           TDNNConfig),
    'resnet34':       (ResNet34,       ResNet34Config),
    'simpleaudiocnn': (SimpleAudioCNN, SimpleAudioCNNConfig),
    'ecapatdnn':      (ECAPATDNN,      ECAPATDNNConfig),
}


REGISTERED_METHODS = {
    'supervised':    (Supervised,   SupervisedConfig),
    'cpc':           (CPC,          CPCConfig),
    'lim':           (LIM,          LIMConfig),
    'simclr':        (SimCLR,       SimCLRConfig),
    'moco':          (MoCo,         MoCoConfig),
    'wmse':          (WMSE,         WMSEConfig),
    'barlow_twins':  (BarlowTwins,  BarlowTwinsConfig),
    'vicreg':        (VICReg,       VICRegConfig),
    'vibcreg':       (VIbCReg,      VIbCRegConfig),
    'byol':          (BYOL,         BYOLConfig),
    'simsiam':       (SimSiam,      SimSiamConfig),
    'dino':          (DINO,         DINOConfig),
    'deepcluster':   (DeepCluster,  DeepClusterConfig),
    'swav':          (SwAV,         SwAVConfig),
    'combiner':      (Combiner,     CombinerConfig),
    'simclr_custom': (SimCLRCustom, SimCLRCustomConfig),
}


def bind_custom_config(data, key, registered_dict):
    type_ = data[key]['type']
    if type_ not in registered_dict.keys():
        raise (
            Exception('{} `{}` not supported'.format(key.capitalize(), type_))
        )

    res = from_dict(registered_dict[type_][1], data[key], DaciteConfig(cast=[Enum]))
    res.__type__ = type_
    return res


def bind_evaluate_tasks_config(data, key, default_config):
    if 'evaluation' not in data.keys() or key not in data['evaluation'].keys():
        return default_config

    tasks = []
    
    for task in data['evaluation'][key]:
        type_ = task['type']
        if type_ not in REGISTERED_EVALUATIONS.keys():
            raise (
                Exception('Evaluation `{}` not supported'.format(type_))
            )

        if type_ in [t.__type__ for t in tasks]:
            raise (
                Exception('Evaluation `{}` already registered'.format(type_))
            )

        res = from_dict(REGISTERED_EVALUATIONS[type_][1], task, DaciteConfig(cast=[Enum]))
        res.__type__ = type_
        tasks.append(res)

    return tasks


def load_config(path, verbose=True):
    data = ruamel.yaml.safe_load(open(path, 'r'))
    config = from_dict(Config, data, DaciteConfig(cast=[Enum]))

    config.evaluation.validation = bind_evaluate_tasks_config(
        data,
        'validation',
        config.evaluation.validation
    )
    config.evaluation.test = bind_evaluate_tasks_config(
        data,
        'test',
        config.evaluation.test
    )
    config.encoder = bind_custom_config(data, 'encoder', REGISTERED_ENCODERS)
    config.method = bind_custom_config(data, 'method', REGISTERED_METHODS)
    config.experiment_name = str(Path(path).parent)
    config.experiment_path = Path(path).parent

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

    # Print config
    if is_main_process() and verbose:
        pp.install_extras(include=['dataclasses'])
        pp.pprint(config)

    return config


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_train_dataloader(config):
    df = pd.read_csv(config.dataset.base_path / config.dataset.train)
    if 'Set' in df.columns: df = df[df['Set'] == 'train']
    files = df['File'].tolist()
    labels = pd.factorize(df[config.dataset.label_key])[0].tolist()
    
    dataset_cls = SSLDataset if config.dataset.ssl else Dataset
    dataset = dataset_cls(
        base_path=config.dataset.base_path,
        files=files,
        labels=labels,
        frame_length=config.dataset.frame_length,
        frame_sampling=config.dataset.frame_sampling,
        num_frames=1,
        augmentation_config=config.dataset.augmentation,
        max_samples=config.dataset.max_samples
    )

    shuffle = True
    sampler = None

    if is_dist_initialized():
        shuffle = False
        sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)

    if config.dataset.sampler and config.dataset.sampler.enable:
        shuffle = False
        sampler = Sampler(
            dataset,
            config.trainer.batch_size,
            config.dataset.sampler
        )
        if is_dist_initialized(): sampler = DistributedSamplerWrapper(sampler)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=config.trainer.batch_size // get_world_size(),
        num_workers=config.dataset.num_workers,
        drop_last=True,
        pin_memory=config.dataset.pin_memory,
        worker_init_fn=seed_dataloader_worker
    )

    return dataloader


def load_model(config):
    encoder_cls = REGISTERED_ENCODERS[config.encoder.__type__][0]
    create_encoder_fn = lambda: encoder_cls(config.encoder)

    model = REGISTERED_METHODS[config.method.__type__][0](
        config.method,
        create_encoder_fn
    )
    return model


def evaluate(model, config, device, validation=False, verbose=True):
    def add_prefix_to_dict_keys(old_dict, prefix):
        new_dict = {}
        for old_key in old_dict.keys():
            new_dict[f'{prefix}{old_key}'] = old_dict[old_key]
        return new_dict

    def evaluate_(tasks, prefix):
        if not validation and prefix == 'val': return {}

        metrics = {}
        for task in tasks:
            evaluation = REGISTERED_EVALUATIONS[task.__type__][0](
                model,
                config,
                task,
                device,
                verbose,
                validation
            )

            task_metrics = evaluation.evaluate()
            task_metrics = add_prefix_to_dict_keys(
                task_metrics,
                prefix=f'{prefix}/{task.__type__}/'
            )
            metrics.update(task_metrics)

        return metrics

    if validation:
        return evaluate_(config.evaluation.validation, 'val')
    
    return evaluate_(config.evaluation.test, 'test')