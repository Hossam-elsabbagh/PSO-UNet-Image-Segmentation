"""Shape and gradient checks for the compact U-Net."""

import torch
from torch import nn

from src.unet import CompactUNet


def test_default_unet_forward_shape_and_backward() -> None:
    model = CompactUNet()
    image = torch.randn(1, 3, 32, 32)
    logits = model(image)
    assert logits.shape == (1, 1, 32, 32)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_dropout_is_configurable_and_confined_to_bottleneck() -> None:
    model = CompactUNet(dropout=0.25)
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout2d)]
    assert len(dropouts) == 1
    assert dropouts[0].p == 0.25
    assert model.get_config()["dropout"] == 0.25


def test_odd_input_size_is_restored() -> None:
    model = CompactUNet(encoder_channels=(2, 4, 8, 16), bottleneck_channels=32)
    assert model(torch.randn(1, 3, 33, 35)).shape == (1, 1, 33, 35)
