from torch import nn
from torch.nn import functional as F



class Conv2dNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups),
            nn.GELU(),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        return self.layer(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            Conv2dNorm(in_ch, in_ch, kernel_size=5, stride=2, padding=2, groups=in_ch),
            Conv2dNorm(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.transition(x)



class TransitionLayerDouble(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.strict = Conv2dNorm(in_ch, out_ch)
        self.large = nn.Sequential(
            Conv2dNorm(in_ch, in_ch, kernel_size=5, stride=2, padding=2, groups=in_ch),
            Conv2dNorm(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.strict(F.interpolate(x, scale_factor=0.5, mode='bilinear')) + self.large(x)



class ResBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Conv2dNorm(channels, channels, kernel_size=5, padding=2, groups=channels),
            Conv2dNorm(channels, channels)
        )
    
    def forward(self, x):
        return x + self.residual(x)



class Stage(nn.Module):
    def __init__(self, channels, repeat) -> None:
        super().__init__()
        modules = [ResBlock(channels) for _ in range(repeat)]
        self.stage = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.stage(x)



class SeparableResNet(nn.Module):
    def __init__(self, num_classes, width_factor=1, depth_factor=1):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dNorm(3, 16*width_factor),
            Stage(16*width_factor, repeat=int(3*depth_factor)),
            TransitionLayer(16*width_factor, 32*width_factor),
            Stage(32*width_factor, repeat=int(7*depth_factor)),
            TransitionLayer(32*width_factor, 64*width_factor),
            Stage(64*width_factor, repeat=int(5*depth_factor)),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64*width_factor, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
