from pathlib import Path
import random
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

import ruamel.yaml
from dacite import from_dict
import prettyprinter as pp

import torch
from torch.utils.data import DataLoader, DistributedSampler

from sslsv.configs import Config
from sslsv.utils.distributed import is_dist_initialized, is_main_process, get_world_size

# Datasets
from sslsv.data.AudioDataset import AudioDataset
from sslsv.data.SiameseAudioDataset import SiameseAudioDataset
from sslsv.data.SupervisedSampler import SupervisedSampler
from sslsv.data.DistributedSamplerWrapper import DistributedSamplerWrapper

# Encoders
from sslsv.encoders.TDNN import TDNN, TDNNConfig
from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.encoders.SimpleAudioCNN import SimpleAudioCNN, SimpleAudioCNNConfig
from sslsv.encoders.ECAPATDNN import ECAPATDNN, ECAPATDNNConfig

# Models
from sslsv.models.Supervised import Supervised, SupervisedConfig
from sslsv.models.Custom import Custom, CustomConfig
from sslsv.models.CPC import CPC, CPCConfig
from sslsv.models.LIM import LIM, LIMConfig
from sslsv.models.SimCLR import SimCLR, SimCLRConfig
from sslsv.models.MoCo import MoCo, MoCoConfig
from sslsv.models.WMSE import WMSE, WMSEConfig
from sslsv.models.BarlowTwins import BarlowTwins, BarlowTwinsConfig
from sslsv.models.VICReg import VICReg, VICRegConfig
from sslsv.models.VIbCReg import VIbCReg, VIbCRegConfig
from sslsv.models.MultiLosses import MultiLosses, MultiLossesConfig
from sslsv.models.BYOL import BYOL, BYOLConfig
from sslsv.models.SimSiam import SimSiam, SimSiamConfig
from sslsv.models.DINO import DINO, DINOConfig
from sslsv.models.DeepCluster import DeepCluster, DeepClusterConfig
from sslsv.models.SwAV import SwAV, SwAVConfig

# Evaluations
from sslsv.evaluation.CosineSVEvaluation import (
    CosineSVEvaluation,
    CosineSVEvaluationTaskConfig
)
from sslsv.evaluation.PLDASVEvaluation import (
    PLDASVEvaluation,
    PLDASVEvaluationTaskConfig
)
from sslsv.evaluation.LogisticRegressionEvaluation import (
    LogisticRegressionEvaluation,
    LogisticRegressionEvaluationTaskConfig
)
from sslsv.evaluation.SVMEvaluation import (
    SVMEvaluation,
    SVMEvaluationTaskConfig
)
from sslsv.evaluation.MLPEvaluation import (
    MLPEvaluation,
    MLPEvaluationTaskConfig
)
from sslsv.evaluation.LinearRegressionEvaluation import (
    LinearRegressionEvaluation,
    LinearRegressionEvaluationTaskConfig
)


REGISTERED_EVALUATIONS = {
    'sv_cosine': (CosineSVEvaluation, CosineSVEvaluationTaskConfig),
    'sv_plda':   (PLDASVEvaluation,   PLDASVEvaluationTaskConfig),
    'svm':       (SVMEvaluation, SVMEvaluationTaskConfig),
    'mlp':       (MLPEvaluation, MLPEvaluationTaskConfig),
    'logistic_regression':  (
        LogisticRegressionEvaluation,
        LogisticRegressionEvaluationTaskConfig
    ),
    'linear_regression':  (
        LinearRegressionEvaluation,
        LinearRegressionEvaluationTaskConfig
    )
}


REGISTERED_ENCODERS = {
    'tdnn':           (TDNN,           TDNNConfig),
    'resnet34':       (ResNet34,       ResNet34Config),
    'simpleaudiocnn': (SimpleAudioCNN, SimpleAudioCNNConfig),
    'ecapatdnn':      (ECAPATDNN,      ECAPATDNNConfig),
}


REGISTERED_MODELS = {
    'supervised':  (Supervised,  SupervisedConfig),
    'custom':      (Custom,      CustomConfig),
    'cpc':         (CPC,         CPCConfig),
    'lim':         (LIM,         LIMConfig),
    'simclr':      (SimCLR,      SimCLRConfig),
    'moco':        (MoCo,        MoCoConfig),
    'wmse':        (WMSE,        WMSEConfig),
    'barlowtwins': (BarlowTwins, BarlowTwinsConfig),
    'vicreg':      (VICReg,      VICRegConfig),
    'vibcreg':     (VIbCReg,     VIbCRegConfig),
    'multilosses': (MultiLosses, MultiLossesConfig),
    'byol':        (BYOL,        BYOLConfig),
    'simsiam':     (SimSiam,     SimSiamConfig),
    'dino':        (DINO,        DINOConfig),
    'deepcluster': (DeepCluster, DeepClusterConfig),
    'swav':        (SwAV,        SwAVConfig),
}


def bind_custom_config(data, key, registered_dict):
    type_ = data[key]['type']
    if type_ not in registered_dict.keys():
        raise (
            Exception('{} `{}` not supported'.format(key.capitalize(), type_))
        )

    res = from_dict(registered_dict[type_][1], data[key])
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

        res = from_dict(REGISTERED_EVALUATIONS[type_][1], task)
        res.__type__ = type_
        tasks.append(res)

    return tasks


def load_config(path, verbose=True):
    load_dotenv()
    
    data = ruamel.yaml.safe_load(open(path, 'r'))
    config = from_dict(Config, data)

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
    config.model = bind_custom_config(data, 'model', REGISTERED_MODELS)
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


def load_train_dataloader(config, nb_labels_per_spk=None):
    df = pd.read_csv(config.data.base_path / config.data.train)
    files = df['File'].tolist()
    labels = pd.factorize(df[config.data.label_key])[0].tolist()
    
    dataset_cls = (
        SiameseAudioDataset if config.data.siamese
        else AudioDataset
    )
    dataset = dataset_cls(
        base_path=config.data.base_path,
        files=files,
        labels=labels,
        frame_length=config.data.frame_length,
        frame_sampling=config.data.frame_sampling,
        num_frames=1,
        augmentation_config=config.data.augmentation,
        max_samples=config.data.max_samples
    )

    shuffle = True
    sampler = None

    if is_dist_initialized():
        shuffle = False
        sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)

    if nb_labels_per_spk:
        shuffle = False
        sampler = SupervisedSampler(
            dataset,
            config.training.batch_size,
            nb_labels_per_spk
        )
        if is_dist_initialized(): sampler = DistributedSamplerWrapper(sampler)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=config.training.batch_size // get_world_size(),
        num_workers=config.data.num_workers,
        drop_last=True,
        pin_memory=config.data.pin_memory,
        worker_init_fn=seed_dataloader_worker
    )

    return dataloader


def load_model(config):
    encoder_cls = REGISTERED_ENCODERS[config.encoder.__type__][0]
    create_encoder_fn = lambda: encoder_cls(config.encoder)

    model = REGISTERED_MODELS[config.model.__type__][0](
        config.model,
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