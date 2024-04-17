import torch
from torch import nn

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.methods.WMSE.WMSE import WMSE, WMSEConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = WMSEConfig()
    method = WMSE(config, create_encoder_fn=lambda: ResNet34(ResNet34Config()))

    assert count_parameters(method) == 2030038

    # Inference
    Z = method(torch.randn(128, 32000))
    assert isinstance(Z, torch.Tensor)
    assert Z.dtype == torch.float32
    assert Z.size() == (128, 512)

    # Training
    Z = method(torch.randn(128, 2, 32000), training=True)
    assert isinstance(Z, tuple)
    assert len(Z) == 2
    for z in Z:
        assert isinstance(z, torch.Tensor)
        assert z.dtype == torch.float32
        assert z.size() == (128, 64)

    # Train step
    loss = method.train_step(Z, step=0)
    metrics = method.step_metrics[0]
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert "train/loss" in metrics
    assert isinstance(metrics["train/loss"], torch.Tensor)
