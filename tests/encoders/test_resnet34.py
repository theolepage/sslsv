import torch

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = ResNet34Config()
    encoder = ResNet34(config)

    assert count_parameters(encoder) == 1437078

    Y = encoder(torch.randn(64, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (64, 512)


def test_default_no_pooling():
    config = ResNet34Config(pooling=False)
    encoder = ResNet34(config)

    assert count_parameters(encoder) == 1682582

    Y = encoder(torch.randn(64, 32000))

    assert isinstance(Y, torch.Tensor)
    assert Y.dtype == torch.float32
    assert Y.size() == (64, 512, 51)
