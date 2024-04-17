import pytest

from pathlib import Path

from sslsv.Config import Config
from sslsv.datasets.Dataset import DatasetConfig
from sslsv.encoders.ResNet34 import ResNet34Config
from sslsv.methods.SimCLR.SimCLR import SimCLRConfig
from sslsv.trainer.Trainer import TrainerConfig
from sslsv.evaluations._BaseEvaluation import BaseEvaluationConfig
from sslsv.utils.helpers import load_config


@pytest.fixture
def default_config() -> Config:
    return Config()


def test_default(default_config: Config):
    assert default_config.model_name == "default"
    assert default_config.model_path == Path("default")
    assert default_config.seed == 1717
    assert default_config.reproducibility is False
    assert default_config.encoder is None
    assert default_config.method is None
    assert default_config.trainer == TrainerConfig()
    assert default_config.dataset == DatasetConfig()
    assert default_config.evaluation == BaseEvaluationConfig()


def test_empty():
    with pytest.raises(KeyError):
        load_config("tests/resources/empty.yml", verbose=False)


def test_no_encoder():
    with pytest.raises(KeyError):
        load_config("tests/resources/no_encoder.yml", verbose=False)


def test_no_method():
    with pytest.raises(KeyError):
        load_config("tests/resources/no_method.yml", verbose=False)


def test_simple():
    config = load_config("tests/resources/simple/config.yml", verbose=False)

    assert config.model_name == "resources/simple"
    assert config.model_path == Path("tests/resources/simple")
    assert config.encoder == ResNet34Config()
    assert config.method == SimCLRConfig()
