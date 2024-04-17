import torch
from torch import nn

from sslsv.encoders.TDNN import TDNN, TDNNConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = TDNNConfig()
    encoder = TDNN(config)

    assert count_parameters(encoder) == 4252564

    Y = encoder(torch.randn(64, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (64, 512)
