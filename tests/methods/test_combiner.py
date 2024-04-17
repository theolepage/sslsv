import torch
from torch import nn

from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.methods.Combiner.Combiner import (
    Combiner,
    CombinerConfig,
    LossItemCombinerConfig,
    LossTypeCombinerEnum,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_default():
    config = CombinerConfig(
        Y_losses=[
            LossItemCombinerConfig(LossTypeCombinerEnum.INFONCE, 1.0),
            LossItemCombinerConfig(LossTypeCombinerEnum.VICREG, 1.0),
        ],
        Z_losses=[LossItemCombinerConfig(LossTypeCombinerEnum.BARLOWTWINS, 1.0)],
    )
    method = Combiner(config, create_encoder_fn=lambda: ResNet34(ResNet34Config()))

    assert count_parameters(method) == 10888598

    # Inference
    Z = method(torch.randn(64, 32000))
    assert isinstance(Z, torch.Tensor)
    assert Z.dtype == torch.float32
    assert Z.size() == (64, 512)

    # Training
    Z = method(torch.randn(64, 2, 32000), training=True)
    assert isinstance(Z, tuple)
    assert len(Z) == 4
    for i, z in enumerate(Z):
        assert isinstance(z, torch.Tensor)
        assert z.dtype == torch.float32
        if i < 2:
            assert z.size() == (64, 512)
        else:
            assert z.size() == (64, 2048)

    # Train step
    loss = method.train_step(Z, step=0)
    metrics = method.step_metrics[0]
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert "train/loss" in metrics
    assert "train/Y_loss" in metrics
    assert "train/Z_loss" in metrics
    assert "train/Y_accuracy" in metrics
    assert "train/Z_accuracy" in metrics
    assert isinstance(metrics["train/loss"], torch.Tensor)
