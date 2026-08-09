import torch
from torch import nn

from sslsv.encoders.ECAPA2 import ECAPA2, ECAPA2Config


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = ECAPA2Config()
    encoder = ECAPA2(config)

    assert count_parameters(encoder) == 27107708

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 192)


def test_default_no_pooling():
    config = ECAPA2Config(pooling=False)
    encoder = ECAPA2(config)

    assert count_parameters(encoder) == 26812796

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 192, 200)
