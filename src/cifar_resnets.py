"""
ResNet for CIFAR-10/100

Implementation of ResNets for CIFAR-10 as described in:
"Deep Residual Learning for Image Recognition" (He et al., 2015)
https://arxiv.org/abs/1512.03385

CIFAR-10 specific architecture:
- Input: 32x32 images
- First layer: 3x3 conv with 16 filters
- Three stages with feature maps: 16, 32, 64
- Global average pooling + fully connected
- Total depth: 6n+2 (e.g., n=3 gives ResNet-20)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNet (no bottleneck)."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet for CIFAR-10/100.
    
    Architecture for 32x32 images:
    - conv1: 16 filters, 3x3, stride 1
    - layer1: n blocks, 16 filters, 32x32 feature maps
    - layer2: n blocks, 32 filters, 16x16 feature maps (stride 2 transition)
    - layer3: n blocks, 64 filters, 8x8 feature maps (stride 2 transition)
    - avgpool: global average pooling to 1x1
    - fc: fully connected layer
    
    Args:
        block: Basic block type
        num_blocks: List of [n, n, n] blocks per stage
        num_classes: Number of output classes
    """
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16
        
        # Initial convolution (for 32x32 input)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20_cifar(num_classes=10):
    """ResNet-20 for CIFAR (n=3, depth=6*3+2=20)"""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32_cifar(num_classes=10):
    """ResNet-32 for CIFAR (n=5, depth=6*5+2=32)"""
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44_cifar(num_classes=10):
    """ResNet-44 for CIFAR (n=7, depth=6*7+2=44)"""
    return ResNetCIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56_cifar(num_classes=10):
    """ResNet-56 for CIFAR (n=9, depth=6*9+2=56)"""
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110_cifar(num_classes=10):
    """ResNet-110 for CIFAR (n=18, depth=6*18+2=110)"""
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes=num_classes)


if __name__ == "__main__":
    # Test all models
    for name, model_fn in [
        ('ResNet-20', resnet20_cifar),
        ('ResNet-32', resnet32_cifar),
        ('ResNet-44', resnet44_cifar),
        ('ResNet-56', resnet56_cifar),
        ('ResNet-110', resnet110_cifar)
    ]:
        model = model_fn(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s}: {num_params:>8,} params, output shape: {y.shape}")