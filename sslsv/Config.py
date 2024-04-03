from dataclasses import dataclass
from pathlib import Path

from sslsv.encoders._BaseEncoder import BaseEncoderConfig
from sslsv.methods._BaseMethod import BaseMethodConfig
from sslsv.evaluations._BaseEvaluation import BaseEvaluationConfig
from sslsv.trainer.Trainer import TrainerConfig
from sslsv.datasets.Dataset import DatasetConfig


@dataclass
class Config:

    experiment_name: str = 'default'
    experiment_path: Path = 'default'
    
    seed: int = 1717
    reproducibility: bool = False

    encoder: BaseEncoderConfig = None
    method: BaseMethodConfig = None
    trainer: TrainerConfig = TrainerConfig()
    dataset: DatasetConfig = DatasetConfig()
    evaluation: BaseEvaluationConfig = BaseEvaluationConfig()