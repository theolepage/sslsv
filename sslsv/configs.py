from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path

from sslsv.evaluation._BaseEvaluation import EvaluationTaskConfig
from sslsv.evaluation.CosineSVEvaluation import CosineSVEvaluationTaskConfig


@dataclass
class TrainingConfig:

    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 300
    tracked_metric: str = 'val/sv_cosine/voxceleb1_test_O/eer'
    tracked_mode: str = 'min'
    optimizer: str = 'adam'
    weight_decay: float = 0
    mixed_precision: bool = False
    init_weights: str = None


@dataclass
class AudioAugmentationConfig:

    enable: bool = True
    rir: bool = True
    musan: bool = True
    musan_nb_iters: int = 1
    musan_noise_snr:  Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr:  Tuple[int, int] = (5, 15)
    musan_noise_num:  Tuple[int, int] = (1, 1)
    musan_speech_num: Tuple[int, int] = (1, 1) # (3, 7)
    musan_music_num:  Tuple[int, int] = (1, 1)


@dataclass
class DataConfig:

    siamese: bool = False
    augmentation: AudioAugmentationConfig = None
    frame_length: int = 32000
    max_samples: int = None
    train: str = 'voxceleb2_train.csv'
    label_key: str = 'Speaker'
    base_path: Path = Path('./data/')
    num_workers: int = 8
    pin_memory: bool = False


@dataclass
class EvaluationConfig:

    batch_size: int = 64
    num_frames: int = 6
    frame_length: int = 32000
    mean_of_features: bool = True

    validation: List[EvaluationTaskConfig] = field(default_factory=lambda: [
        CosineSVEvaluationTaskConfig(__type__='sv_cosine')
    ])

    test: List[EvaluationTaskConfig] = field(default_factory=lambda: [
        CosineSVEvaluationTaskConfig(__type__='sv_cosine')
    ])


@dataclass
class EncoderConfig:

    __type__: str = None


@dataclass
class ModelConfig:

    __type__: str = None


@dataclass
class Config:

    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    encoder: EncoderConfig = EncoderConfig()
    model: ModelConfig = ModelConfig()
    name: str = 'test'
    seed: int = 1717
    reproducibility: bool = False
    wandb_id: str = None
    wandb_project: str = 'sslsv'
