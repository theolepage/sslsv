from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:

    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 300
    tracked_metric: str = 'test_eer'
    tracked_mode: str = 'min'
    optimizer: str = 'adam'
    weight_decay: float = 0
    mixed_precision: bool = False


@dataclass
class WavAugmentConfig:

    enable: bool = True
    rir: bool = True
    musan: bool = True
    musan_noise_snr: Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr: Tuple[int, int] = (5, 15)


@dataclass
class DataConfig:

    siamese: bool = False
    wav_augment: WavAugmentConfig = None
    frame_length: int = 32000
    max_samples: int = None
    train: str = './data/voxceleb2_train_list'
    trials: str = './data/trials'
    base_path: str = './data/'
    enable_cache: bool = False
    num_workers: int = 8
    pin_memory: bool = False


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
    encoder: EncoderConfig = EncoderConfig()
    model: ModelConfig = ModelConfig()
    name: str = 'test'
    seed: int = 1717
    reproducibility: bool = False
    wandb_id: str = None
