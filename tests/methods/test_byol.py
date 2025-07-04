import torch
from torch import nn

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.methods.BYOL.BYOL import BYOL, BYOLConfig

from tests.utils import add_dummy_trainer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = BYOLConfig()
    method = BYOL(config, create_encoder_fn=lambda: ResNet34(ResNet34Config()))

    add_dummy_trainer(method)

    assert count_parameters(method) == 23494550

    # Inference
    Z = method(torch.randn(64, 32000))
    assert isinstance(Z, torch.Tensor)
    assert Z.dtype == torch.float32
    assert Z.size() == (64, 512)

    # Training
    Z = method(torch.randn(64, 2, 32000), training=True)
    assert isinstance(Z, tuple)
    assert len(Z) == 4
    for z in Z:
        assert isinstance(z, torch.Tensor)
        assert z.dtype == torch.float32
        assert z.size() == (64, 256)

    # Train step
    loss = method.train_step(Z, step=0)
    metrics = method.step_metrics
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert "train/loss" in metrics
    assert "train/tau" in metrics
    assert isinstance(metrics["train/loss"], torch.Tensor)
