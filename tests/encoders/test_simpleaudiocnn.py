import torch
from torch import nn

from sslsv.encoders.SimpleAudioCNN import SimpleAudioCNN, SimpleAudioCNNConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = SimpleAudioCNNConfig()
    encoder = SimpleAudioCNN(config)

    assert count_parameters(encoder) == 5253120

    Y = encoder(torch.randn(64, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (64, 512, 200)
