import torch
from torch import nn

from sslsv.encoders.ResNet293 import ResNet293, ResNet293Config, PoolingModeEnum


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = ResNet293Config()
    encoder = ResNet293(config)

    assert count_parameters(encoder) == 28626016

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 256)


def test_default_astp():
    config = ResNet293Config(pooling_mode=PoolingModeEnum.ASTP)
    encoder = ResNet293(config)

    assert count_parameters(encoder) == 33879264

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 256)


def test_default_no_pooling():
    config = ResNet293Config(pooling=False)
    encoder = ResNet293(config)

    assert count_parameters(encoder) == 26004576

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 256, 26)
