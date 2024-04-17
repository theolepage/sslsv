import torch
from torch import nn

from sslsv.encoders.ECAPATDNN import ECAPATDNN, ECAPATDNNConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = ECAPATDNNConfig()
    encoder = ECAPATDNN(config)

    assert count_parameters(encoder) == 7075008

    Y = encoder(torch.randn(64, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (64, 512)
