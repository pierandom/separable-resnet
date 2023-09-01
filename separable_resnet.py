import torch
from torch import nn


class Conv2dNorm(nn.Sequential):
    def __init__(
        self, channels_in, channels_out, kernel_size=1, stride=1, padding=0, groups=1
    ):
        super().__init__(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.GELU(),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1) -> None:
        super().__init__(
            Conv2dNorm(
                channels_in,
                channels_in,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                groups=channels_in,
            ),
            Conv2dNorm(channels_in, channels_out),
        )


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size) -> None:
        super().__init__()
        self.residual = SeparableConv(channels, channels, kernel_size)
        self.register_buffer("sqrt_2", torch.tensor(2).sqrt(), persistent=False)

    def forward(self, x):
        return (x + self.residual(x)) / self.sqrt_2


class Stage(nn.Sequential):
    def __init__(self, channels, kernel_size, repeat) -> None:
        super().__init__(*[ResBlock(channels, kernel_size) for _ in range(repeat)])


class SeparableResNet(nn.Sequential):
    def __init__(self, num_classes, kernel_size=5, width_factor=1, depth_factor=1):
        super().__init__(
            Conv2dNorm(3, 16 * width_factor, kernel_size=1),
            Stage(16 * width_factor, kernel_size, repeat=depth_factor),
            SeparableConv(16 * width_factor, 32 * width_factor, kernel_size, stride=2),
            Stage(32 * width_factor, kernel_size, repeat=depth_factor),
            SeparableConv(32 * width_factor, 64 * width_factor, kernel_size, stride=2),
            Stage(64 * width_factor, kernel_size, repeat=depth_factor),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64 * width_factor, num_classes),
        )
