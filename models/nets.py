import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """用于 CIFAR-10 的轻量 CNN，约 1.2M 参数"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
import torch.nn as nn
import math

class MobileNetV2Cifar(nn.Module):
    """轻量 MobileNetV2 用于 CIFAR-100 (约 2.3M 参数)"""
    def __init__(self, num_classes=100, width_mult=0.5):
        super(MobileNetV2Cifar, self).__init__()
        # 简化版 inverted residual blocks
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        def conv_1x1_bn(inp, oup):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        class InvertedResidual(nn.Module):
            def __init__(self, inp, oup, stride, expand_ratio):
                super(InvertedResidual, self).__init__()
                self.stride = stride
                hidden_dim = int(inp * expand_ratio)
                self.use_res_connect = self.stride == 1 and inp == oup
                layers = []
                if expand_ratio != 1:
                    layers.append(conv_1x1_bn(inp, hidden_dim))
                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                ])
                self.conv = nn.Sequential(*layers)
            def forward(self, x):
                if self.use_res_connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)

        # 构建网络
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 1)
        # 一系列 inverted residual blocks (简化版，少量层)
        self.blocks = nn.Sequential(
            InvertedResidual(input_channel, int(16 * width_mult), 1, 1),
            InvertedResidual(int(16 * width_mult), int(24 * width_mult), 2, 6),
            InvertedResidual(int(24 * width_mult), int(32 * width_mult), 2, 6),
        )
        last_channel = int(128 * width_mult) if width_mult > 1.0 else 128
        self.conv2 = conv_1x1_bn(int(32 * width_mult), last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x