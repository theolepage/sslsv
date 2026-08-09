import torch
from torch import nn

from sslsv.encoders.ReDimNet2 import ReDimNet2, ReDimNet2Config


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = ReDimNet2Config()
    encoder = ReDimNet2(config)

    assert count_parameters(encoder) == 12454688

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 192)


def test_default_no_pooling():
    config = ReDimNet2Config(pooling=False)
    encoder = ReDimNet2(config)

    assert count_parameters(encoder) == 11025216

    Y = encoder(torch.randn(2, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (2, 192, 200)
