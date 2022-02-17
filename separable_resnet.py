from torch import nn



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



class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            Conv2dNorm(in_ch, in_ch, kernel_size, stride, padding=kernel_size//2, groups=in_ch),
            Conv2dNorm(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.layer(x)



class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size) -> None:
        super().__init__()
        self.residual = SeparableConv(channels, channels, kernel_size)
    
    def forward(self, x):
        return x + self.residual(x)



class Stage(nn.Module):
    def __init__(self, channels, kernel_size, repeat) -> None:
        super().__init__()
        modules = [ResBlock(channels, kernel_size) for _ in range(repeat)]
        self.stage = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.stage(x)



class SeparableResNet(nn.Module):
    def __init__(self, num_classes, kernel_size=5, width_factor=1, depth_factor=1):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dNorm(3, 16*width_factor, kernel_size=1),
            Stage(16*width_factor, kernel_size, repeat=(3*depth_factor)),
            SeparableConv(16*width_factor, 32*width_factor, kernel_size, stride=2),
            Stage(32*width_factor, kernel_size, repeat=(2*depth_factor)),
            SeparableConv(32*width_factor, 64*width_factor, kernel_size, stride=2),
            Stage(64*width_factor, kernel_size, repeat=(1*depth_factor)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64*width_factor, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
