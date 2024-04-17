import torch
from torch import nn

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.methods.CPC.CPC import CPC, CPCConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = CPCConfig()
    method = CPC(
        config, create_encoder_fn=lambda: ResNet34(ResNet34Config(pooling=False))
    )

    assert count_parameters(method) == 2800278

    # Inference
    Z = method(torch.randn(64, 32000))
    assert isinstance(Z, torch.Tensor)
    assert Z.dtype == torch.float32
    assert Z.size() == (64, 256)

    # Training
    Z = method(torch.randn(64, 2, 32000), training=True)
    assert isinstance(Z, tuple)
    assert len(Z) == 4
    for i, z in enumerate(Z):
        if i < 2:
            assert isinstance(z, torch.Tensor)
            assert z.dtype == torch.float32
            assert z.size() == (64, 512, 51)
        else:
            assert z is None

    # Train step
    loss = method.train_step(Z, step=0)
    metrics = method.step_metrics[0]
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert "train/loss" in metrics
    assert isinstance(metrics["train/loss"], torch.Tensor)
