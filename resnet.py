from torch import nn
from torch.nn import functional as F

def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False))
            self.shortcut.add_module("bn", nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y)
        return y
    


class ResNet(nn.Module):
    def __init__(self, depth, n_classes=10, base_channels=16):
        super().__init__()

        n_blocks_per_stage = (depth-2) // 6
        n_channels = [
            base_channels,
            2*base_channels,
            4*base_channels
        ]

        self.conv = nn.Conv2d(
            3,
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, stride=2)

        self.fc = nn.Linear(n_channels[2], n_classes)

        self.apply(init_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"block{index+1}"
            if index == 0:
                stage.add_module(block_name,
                    BasicBlock(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name,
                    BasicBlock(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet32():
    return ResNet(depth=32)
