from typing import Any, Dict, List, Tuple
from enum import Enum

from pathlib import Path
import random
import os

import numpy as np
import pandas as pd

from ruamel.yaml import YAML
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
from sslsv.methods._BaseMethod import BaseMethod
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
from sslsv.methods.SimCLRMargins.SimCLRMargins import SimCLRMargins, SimCLRMarginsConfig
from sslsv.methods.MoCoMargins.MoCoMargins import MoCoMargins, MoCoMarginsConfig

# Evaluations
from sslsv.evaluations._BaseEvaluation import EvaluationTaskConfig
from sslsv.evaluations.CosineSVEvaluation import (
    CosineSVEvaluation,
    CosineSVEvaluationTaskConfig,
)
from sslsv.evaluations.PLDASVEvaluation import (
    PLDASVEvaluation,
    PLDASVEvaluationTaskConfig,
)
from sslsv.evaluations.ClassificationEvaluation import (
    ClassificationEvaluation,
    ClassificationEvaluationTaskConfig,
)


LOGO = """
         _
 ___ ___| |_____   __
/ __/ __| / __\ \ / /
\__ \__ \ \__ \\\\ V /
|___/___/_|___/ \_/
"""

REGISTERED_EVALUATIONS = {
    "sv_cosine": (CosineSVEvaluation, CosineSVEvaluationTaskConfig),
    "sv_plda": (PLDASVEvaluation, PLDASVEvaluationTaskConfig),
    "classification": (ClassificationEvaluation, ClassificationEvaluationTaskConfig),
}


REGISTERED_ENCODERS = {
    "tdnn": (TDNN, TDNNConfig),
    "resnet34": (ResNet34, ResNet34Config),
    "simpleaudiocnn": (SimpleAudioCNN, SimpleAudioCNNConfig),
    "ecapatdnn": (ECAPATDNN, ECAPATDNNConfig),
}


REGISTERED_METHODS = {
    "supervised": (Supervised, SupervisedConfig),
    "cpc": (CPC, CPCConfig),
    "lim": (LIM, LIMConfig),
    "simclr": (SimCLR, SimCLRConfig),
    "moco": (MoCo, MoCoConfig),
    "wmse": (WMSE, WMSEConfig),
    "barlow_twins": (BarlowTwins, BarlowTwinsConfig),
    "vicreg": (VICReg, VICRegConfig),
    "vibcreg": (VIbCReg, VIbCRegConfig),
    "byol": (BYOL, BYOLConfig),
    "simsiam": (SimSiam, SimSiamConfig),
    "dino": (DINO, DINOConfig),
    "deepcluster": (DeepCluster, DeepClusterConfig),
    "swav": (SwAV, SwAVConfig),
    "combiner": (Combiner, CombinerConfig),
    "simclr_margins": (SimCLRMargins, SimCLRMarginsConfig),
    "moco_margins": (MoCoMargins, MoCoMarginsConfig),
}


def show_logo():
    """
    Print sslsv logo.

    Returns:
        None
    """
    print(LOGO)
    print()


def _bind_config(
    data: Dict[str, Any],
    key: str,
    registered_dict: Dict[str, Tuple[Any, Any]],
) -> Any:
    """
    Create a config dataclass, among a set of supported dataclasses (`registered_dict`:
    REGISTERED_METHODS or REGISTERED_ENCODERS), from the values of a dictionary (`data`).

    Args:
        data (Dict[str, Any]): Configuration data in dictionary format.
        key (str): Key to access the dataclass to create.
        registered_dict (Dict[str, Tuple[Any, Any]]): Dictionary containing supported dataclasses.

    Returns:
        Any: Configuration dataclass.

    Raises:
        Exception: If the specified configuration type is not supported.
    """
    type_ = data[key]["type"]
    if type_ not in registered_dict.keys():
        raise Exception("{} `{}` not supported".format(key.capitalize(), type_))

    res = from_dict(registered_dict[type_][1], data[key], DaciteConfig(cast=[Enum]))
    res.__type__ = type_
    return res


def _bind_evaluate_tasks_config(
    data: Dict[str, Any],
    key: str,
    default_config: List[EvaluationTaskConfig],
) -> List[EvaluationTaskConfig]:
    """
    Create a list of evaluation task config dataclasses, among a set of supported dataclasses
    (`REGISTERED_EVALUATIONS`), from the values of a dictionary (`data`).

    Args:
        data (Dict[str, Any]): Configuration data in dictionary format.
        key (str): Key to access the evaluation task configurations.
        default_config (List[EvaluationTaskConfig]): Default list of evaluation task configurations.

    Returns:
        List[EvaluationTaskConfig]: List of evaluation task configurations.

    Raises:
        Exception: If the evaluation task type is not supported.
        Exception: If an evaluation task of the same type is already registered.
    """
    if "evaluation" not in data.keys() or key not in data["evaluation"].keys():
        return default_config

    tasks = []

    for task in data["evaluation"][key]:
        type_ = task["type"]
        if type_ not in REGISTERED_EVALUATIONS.keys():
            raise Exception("Evaluation `{}` not supported".format(type_))

        if type_ in [t.__type__ for t in tasks]:
            raise Exception("Evaluation `{}` already registered".format(type_))

        res = from_dict(
            REGISTERED_EVALUATIONS[type_][1], task, DaciteConfig(cast=[Enum])
        )
        res.__type__ = type_
        tasks.append(res)

    return tasks


def load_config(path: str, verbose: bool = True) -> Config:
    """
    Load a configuration file and create a Config object.

    Args:
        path (str): Path to the configuration file.
        verbose (bool): Verbosity flag. Defaults to True.

    Returns:
        Config: Config object containing the loaded configuration.
    """
    data = YAML(typ="safe", pure=True).load(open(path, "r"))
    config = from_dict(Config, data, DaciteConfig(cast=[Enum]))

    config.evaluation.validation = _bind_evaluate_tasks_config(
        data, "validation", config.evaluation.validation
    )
    config.evaluation.test = _bind_evaluate_tasks_config(
        data, "test", config.evaluation.test
    )
    config.encoder = _bind_config(data, "encoder", REGISTERED_ENCODERS)
    config.method = _bind_config(data, "method", REGISTERED_METHODS)

    path = Path(path)
    config.model_name = str(path.parent.relative_to(path.parts[0]))
    config.model_path = path.parent
    config.model_ckpt_path = config.model_path / "checkpoints"

    # Reproducibility / performance
    torch.backends.cudnn.benchmark = not config.reproducibility
    torch.backends.cudnn.deterministic = config.reproducibility
    torch.use_deterministic_algorithms(config.reproducibility)

    # Set seed
    seed = config.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Print logo and config
    if is_main_process() and verbose:
        show_logo()
        pp.install_extras(include=["dataclasses"])
        pp.pprint(config)

    return config


def seed_dataloader_worker(worker_id: int):
    """
    Set the seed for a PyTorch DataLoader worker for reproducibility.

    Args:
        worker_id (int): ID of the DataLoader worker.

    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_train_dataloader(config: Config) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for training data.

    Args:
        config (Config): Global configuration.

    Returns:
        torch.utils.data.DataLoader: DataLoader for training data.
    """
    df = pd.read_csv(config.dataset.base_path / config.dataset.train)
    if "Set" in df.columns:
        df = df[df["Set"] == "train"]
    files = df["File"].tolist()
    labels = pd.factorize(df[config.dataset.label_key])[0].tolist()

    dataset_cls = SSLDataset if config.dataset.ssl else Dataset
    dataset = dataset_cls(config.dataset, files, labels, num_frames=1)

    shuffle = True
    sampler = None

    if is_dist_initialized():
        shuffle = False
        sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)

    if config.dataset.sampler and config.dataset.sampler.enable:
        shuffle = False
        sampler = Sampler(
            dataset.labels,
            config.trainer.batch_size,
            config.dataset.sampler,
            seed=config.seed,
        )
        if is_dist_initialized():
            sampler = DistributedSamplerWrapper(sampler)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=config.trainer.batch_size // get_world_size(),
        num_workers=config.dataset.num_workers,
        drop_last=True,
        pin_memory=config.dataset.pin_memory,
        worker_init_fn=seed_dataloader_worker,
    )

    return dataloader


def load_model(config: Config) -> BaseMethod:
    """
    Load a model by instantiating a method and its encoder.

    Args:
        config (Config): Global configuration.

    Returns:
        BaseMethod: Instance of the resulting method.
    """
    encoder_cls = REGISTERED_ENCODERS[config.encoder.__type__][0]
    create_encoder_fn = lambda: encoder_cls(config.encoder)

    model = REGISTERED_METHODS[config.method.__type__][0](
        config.method, create_encoder_fn
    )
    return model


def evaluate(
    model: BaseMethod,
    config: Config,
    device: torch.device,
    validation: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model.

    Args:
        model (BaseMethod): Model to evaluate.
        config (Config): Global configuration.
        device (torch.device): Device on which tensors will be allocated.
        validation (bool): Flag to indicate if validation evaluation is to be performed.
            Otherwise test evaluation is to be performed. Defaults to False.
        verbose (bool): Verbosity flag. Defaults to True.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """

    def add_prefix_to_dict_keys(
        old_dict: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """
        Add a prefix to the keys of a dictionary.

        Args:
            old_dict (Dict[str, Any]): Original dictionary.
            prefix (str): Prefix to be added to the keys.

        Returns:
            Dict[str, Any]: Dictionary with keys that have the specified prefix added.
        """
        new_dict = {}
        for old_key in old_dict.keys():
            new_dict[f"{prefix}{old_key}"] = old_dict[old_key]
        return new_dict

    def evaluate_(tasks: List[EvaluationTaskConfig], prefix: str) -> Dict[str, float]:
        """
        Evaluate a list of tasks and return a dictionary of metrics.

        Args:
            tasks (List[EvaluationTaskConfig]): List of evaluation tasks to be performed.
            prefix (str): String to be added as a prefix to each metric key.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if not validation and prefix == "val":
            return {}

        metrics = {}
        for task in tasks:
            if prefix == "test":  # FIXME
                task.trials = [
                    "voxceleb1_test_O",
                    "voxceleb1_test_H",
                    "voxceleb1_test_E",
                ]

            evaluation = REGISTERED_EVALUATIONS[task.__type__][0](
                model,
                config,
                task,
                device,
                verbose,
                validation,
            )

            task_metrics = evaluation.evaluate()
            task_metrics = add_prefix_to_dict_keys(
                task_metrics, prefix=f"{prefix}/{task.__type__}/"
            )
            metrics.update(task_metrics)

        return metrics

    if validation:
        return evaluate_(config.evaluation.validation, "val")

    return evaluate_(config.evaluation.test, "test")
