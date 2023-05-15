from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path


@dataclass
class TrainingConfig:

    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 300
    tracked_metric: str = 'val/eer'
    tracked_mode: str = 'min'
    optimizer: str = 'adam'
    weight_decay: float = 0
    mixed_precision: bool = False


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
    train: str = 'voxceleb2_train'
    val: str = 'voxceleb1_test_O'
    test: List[str] = field(default_factory=lambda: [
        'voxceleb1_test_O',
        # 'voxceleb1_test_H',
        # 'voxceleb1_test_E',
        # 'voxsrc2021_val',
        # 'voices2019_dev'
    ])
    base_path: Path = Path('./data/')
    enable_cache: bool = False
    num_workers: int = 8
    pin_memory: bool = False


@dataclass
class EvaluateConfig:

    method: str = 'cosine'
    batch_size: int = 64
    num_frames: int = 6
    frame_length: int = 32000
    mean_of_features: bool = True

    metrics: List[str] = field(default_factory=lambda: [
        'eer',
        'mindcf',
        # 'actdcf',
        # 'cllr',
        # 'avgrprec'
    ])

    score_norm: str = None#'s-norm'
    score_norm_cohort_size: int = 20000

    mindcf_p_target: float = 0.01
    mindcf_c_miss: float = 1
    mindcf_c_fa: float = 1


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
    evaluate: EvaluateConfig = EvaluateConfig()
    encoder: EncoderConfig = EncoderConfig()
    model: ModelConfig = ModelConfig()
    name: str = 'test'
    seed: int = 1717
    reproducibility: bool = False
    wandb_id: str = None
    wandb_project: str = 'sslsv'
