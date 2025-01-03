import torch
from torch import nn

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.methods.DINO.DINO import DINO, DINOConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = DINOConfig()
    method = DINO(config, create_encoder_fn=lambda: ResNet34(ResNet34Config()))

    assert count_parameters(method) == 23994006

    # Inference
    Z = method(torch.randn(64, 32000))
    assert isinstance(Z, torch.Tensor)
    assert Z.dtype == torch.float32
    assert Z.size() == (64, 512)

    # Training
    Z = method(torch.randn(64, 6, 32000), training=True)
    assert isinstance(Z, tuple)
    assert isinstance(Z[0], torch.Tensor)
    assert isinstance(Z[1], torch.Tensor)
    assert Z[0].dtype == torch.float32
    assert Z[1].dtype == torch.float32
    assert Z[0].size() == (64 * 6, 65536)
    assert Z[1].size() == (64 * 2, 65536)

    # Train step
    loss = method.train_step(Z, step=0)
    metrics = method.step_metrics
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert "train/loss" in metrics
    assert isinstance(metrics["train/loss"], torch.Tensor)
