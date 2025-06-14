from dataclasses import dataclass, field

from pathlib import Path

from sslsv.encoders._BaseEncoder import BaseEncoderConfig
from sslsv.methods._BaseMethod import BaseMethodConfig
from sslsv.evaluations._BaseEvaluation import BaseEvaluationConfig
from sslsv.trainer.Trainer import TrainerConfig
from sslsv.datasets.Dataset import DatasetConfig


@dataclass
class Config:
    """
    Global configuration.

    Attributes:
        model_name (str): Name of the model.
        model_path (Path): Path to the model directory.
        seed (int): Seed for reproducibility.
        reproducibility (bool): Whether or not to enable reproducibility mode.
        encoder (BaseEncoderConfig): Encoder configuration.
        method (BaseMethodConfig): Method configuration.
        trainer (TrainerConfig): Trainer configuration.
        dataset (DatasetConfig): Dataset configuration.
        evaluation (BaseEvaluationConfig): Evaluation configuration.
    """

    model_name: str = "default"
    model_path: Path = Path("default")

    seed: int = 1717
    reproducibility: bool = False

    encoder: BaseEncoderConfig = None
    method: BaseMethodConfig = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: BaseEvaluationConfig = field(default_factory=BaseEvaluationConfig)
