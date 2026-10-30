"""Compact U-Net for binary pet segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two padded 3x3 convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CompactUNet(nn.Module):
    """Four-level U-Net returning one raw logit per pixel.

    Dropout is applied only after the bottleneck double convolution. The same
    model class is used for the fixed baseline and every PSO candidate.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        bottleneck_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4 or any(ch <= 0 for ch in encoder_channels):
            raise ValueError("encoder_channels must contain four positive integers")
        if input_channels <= 0 or output_channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channel counts must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.model_config = {
            "input_channels": int(input_channels),
            "output_channels": int(output_channels),
            "encoder_channels": tuple(int(ch) for ch in encoder_channels),
            "bottleneck_channels": int(bottleneck_channels),
            "dropout": float(dropout),
        }
        channels = self.model_config["encoder_channels"]

        self.encoder = nn.ModuleList()
        current_channels = input_channels
        for channels_out in channels:
            self.encoder.append(DoubleConv(current_channels, channels_out))
            current_channels = channels_out

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = DoubleConv(current_channels, bottleneck_channels)
        self.bottleneck_dropout = nn.Dropout2d(p=dropout)

        self.decoder = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.decoder.append(DoubleConv(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels
        self.output_conv = nn.Conv2d(current_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck_dropout(self.bottleneck(x))
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((skip, x), dim=1))
        return self.output_conv(x)

    def get_config(self) -> dict[str, object]:
        """Return constructor arguments suitable for checkpoint recreation."""
        return dict(self.model_config)
