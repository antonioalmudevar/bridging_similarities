"""
DenseNet for CIFAR-10/100

Implementation of DenseNets for CIFAR-10/100 as described in:
"Densely Connected Convolutional Networks" (Huang et al., 2017)
https://arxiv.org/abs/1608.06993

CIFAR-10 specific architecture:
- Input: 32x32 images
- First layer: 3x3 conv with 2*growth_rate filters
- Three dense blocks with compression
- Growth rate k (typically 12 or 24)
- Total depth: determined by layers per block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic DenseNet block: BN-ReLU-Conv(3x3)."""
    
    def __init__(self, in_planes, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, growth_rate, kernel_size=3, 
                               padding=1, bias=False)
        self.growth_rate = growth_rate
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat([out, x], 1)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck DenseNet block: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)."""
    
    def __init__(self, in_planes, growth_rate):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, expansion * growth_rate, 
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * growth_rate)
        self.conv2 = nn.Conv2d(expansion * growth_rate, growth_rate, 
                               kernel_size=3, padding=1, bias=False)
        self.growth_rate = growth_rate
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class TransitionBlock(nn.Module):
    """Transition layer: BN-ReLU-Conv(1x1)-AvgPool(2x2).
    
    Reduces spatial dimensions by 2 and compresses channels.
    """
    
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNetCIFAR(nn.Module):
    """
    DenseNet for CIFAR-10/100.
    
    Architecture for 32x32 images:
    - conv1: 2*growth_rate filters, 3x3, no stride
    - dense_block1: L layers, 32x32 feature maps
    - transition1: compression, 32->16
    - dense_block2: L layers, 16x16 feature maps  
    - transition2: compression, 16->8
    - dense_block3: L layers, 8x8 feature maps
    - bn-relu-avgpool: global average pooling to 1x1
    - fc: fully connected layer
    
    Args:
        block: BasicBlock or BottleneckBlock
        num_blocks: List of [L, L, L] layers per dense block
        growth_rate: Growth rate k (number of filters per layer)
        reduction: Compression factor at transitions (0 < reduction <= 1)
        num_classes: Number of output classes
    """
    
    def __init__(self, block, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNetCIFAR, self).__init__()
        self.growth_rate = growth_rate
        
        # Initial number of channels
        num_planes = 2 * growth_rate
        
        # First convolution (for 32x32 input, no downsampling)
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks and transitions
        self.dense1 = self._make_dense_block(block, num_planes, num_blocks[0])
        num_planes += num_blocks[0] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans1 = TransitionBlock(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense2 = self._make_dense_block(block, num_planes, num_blocks[1])
        num_planes += num_blocks[1] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans2 = TransitionBlock(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense3 = self._make_dense_block(block, num_planes, num_blocks[2])
        num_planes += num_blocks[2] * growth_rate
        
        # Final batch norm
        self.bn = nn.BatchNorm2d(num_planes)
        
        # Classifier
        self.fc = nn.Linear(num_planes, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _make_dense_block(self, block, in_planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def densenet40_cifar(num_classes=10):
    """
    DenseNet-40 for CIFAR (L=12, k=12).
    
    Depth = 1 + 3*L + 2*3 + 1 = 1 + 3*12 + 6 + 1 = 44 layers
    But conventionally called DenseNet-40 (L=12 per block, 3 blocks, plus other layers)
    
    Growth rate k=12, compression=0.5
    """
    return DenseNetCIFAR(BasicBlock, [12, 12, 12], growth_rate=12, 
                         reduction=0.5, num_classes=num_classes)


def densenet100_cifar(num_classes=10):
    """
    DenseNet-100 for CIFAR (L=32, k=12).
    
    Much deeper version.
    Growth rate k=12, compression=0.5
    """
    return DenseNetCIFAR(BasicBlock, [32, 32, 32], growth_rate=12, 
                         reduction=0.5, num_classes=num_classes)


def densenet_bc_100_cifar(num_classes=10):
    """
    DenseNet-BC-100 for CIFAR (L=16, k=12, with bottleneck).
    
    Uses bottleneck blocks (BC = Bottleneck + Compression).
    Growth rate k=12, compression=0.5
    """
    return DenseNetCIFAR(BottleneckBlock, [16, 16, 16], growth_rate=12, 
                         reduction=0.5, num_classes=num_classes)


def densenet_bc_250_cifar(num_classes=10):
    """
    DenseNet-BC-250 for CIFAR (L=41, k=12, with bottleneck).
    
    Very deep version with bottleneck.
    Growth rate k=12, compression=0.5
    """
    return DenseNetCIFAR(BottleneckBlock, [41, 41, 41], growth_rate=12, 
                         reduction=0.5, num_classes=num_classes)


def densenet_bc_190_cifar(num_classes=10):
    """
    DenseNet-BC-190 for CIFAR (L=31, k=12, with bottleneck).
    
    Deep version with bottleneck.
    Growth rate k=12, compression=0.5
    """
    return DenseNetCIFAR(BottleneckBlock, [31, 31, 31], growth_rate=12, 
                         reduction=0.5, num_classes=num_classes)


if __name__ == "__main__":
    # Test all models
    models = [
        ('DenseNet-40', densenet40_cifar),
        ('DenseNet-100', densenet100_cifar),
        ('DenseNet-BC-100', densenet_bc_100_cifar),
        ('DenseNet-BC-190', densenet_bc_190_cifar),
        ('DenseNet-BC-250', densenet_bc_250_cifar),
    ]
    
    print("Testing DenseNet models for CIFAR-10 (32x32)\n")
    print("=" * 70)
    
    for name, model_fn in models:
        model = model_fn(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s}: {num_params:>8,} params, output: {y.shape}")
    
    print("=" * 70)
    print("\nAll tests passed!")