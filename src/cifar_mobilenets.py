"""
MobileNetV3 and ShuffleNetV2 adapted for CIFAR-10/100 (32x32 images)

These are simplified versions designed for 32x32 input, not ImageNet's 224x224.
Key modifications:
- First conv stride 1 (not 2) to preserve spatial resolution
- Removed early downsampling layers
- Adjusted channel dimensions for smaller images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    """Hard Swish activation for MobileNetV3."""
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation for MobileNetV3."""
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
    
    def forward(self, x):
        out = self.pool(x)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out


class MobileNetV3Block(nn.Module):
    """MobileNetV3 inverted residual block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se=False, nl='RE'):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        # Activation
        if nl == 'RE':
            activation = nn.ReLU(inplace=True)
        elif nl == 'HS':
            activation = HardSwish()
        else:
            raise ValueError(f"Unknown activation: {nl}")
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])
        
        # SE
        if se:
            layers.append(SEBlock(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3_CIFAR(nn.Module):
    """
    MobileNetV3-Small adapted for CIFAR-10/100 (32x32 images).
    
    Key changes from ImageNet version:
    - First conv stride 1 (not 2) 
    - Fewer downsampling operations
    - Smaller channel dimensions
    - Designed for 32x32 input
    """
    
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        
        # [k, exp, c, se, nl, s]
        # k: kernel size
        # exp: expansion ratio
        # c: output channels
        # se: squeeze-excitation
        # nl: nonlinearity (RE=ReLU, HS=HardSwish)
        # s: stride
        
        # Modified for CIFAR: reduced downsampling
        self.cfg = [
            # Stage 1: 32x32
            [3, 1,  16, True,  'RE', 1],
            [3, 4,  24, False, 'RE', 2],  # -> 16x16
            [3, 3,  24, False, 'RE', 1],
            # Stage 2: 16x16
            [5, 3,  40, True,  'HS', 2],  # -> 8x8
            [5, 3,  40, True,  'HS', 1],
            [5, 3,  40, True,  'HS', 1],
            # Stage 3: 8x8
            [5, 6,  48, True,  'HS', 1],
            [5, 6,  48, True,  'HS', 1],
            # Stage 4: 8x8 (no downsample, keep 8x8)
            [5, 6,  96, True,  'HS', 2],  # -> 4x4
            [5, 6,  96, True,  'HS', 1],
            [5, 6,  96, True,  'HS', 1],
        ]
        
        input_channels = 16
        
        # First layer
        self.features = [
            nn.Conv2d(3, input_channels, 3, stride=1, padding=1, bias=False),  # stride=1 for CIFAR!
            nn.BatchNorm2d(input_channels),
            HardSwish()
        ]
        
        # Build inverted residual blocks
        for k, exp, c, se, nl, s in self.cfg:
            output_channels = int(c * width_mult)
            self.features.append(
                MobileNetV3Block(input_channels, output_channels, k, s, exp, se, nl)
            )
            input_channels = output_channels
        
        # Last conv
        last_channels = int(576 * width_mult)
        self.features.extend([
            nn.Conv2d(input_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            HardSwish()
        ])
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, 1024),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ShuffleNetV2_CIFAR(nn.Module):
    """
    ShuffleNetV2 adapted for CIFAR-10/100 (32x32 images).
    
    Key changes from ImageNet version:
    - First conv stride 1 (not 2)
    - Fewer stages
    - Adjusted for 32x32 spatial dimensions
    """
    
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        
        # Channel configurations for different width multipliers
        if width_mult == 0.5:
            channels = [24, 48, 96, 192]
        elif width_mult == 1.0:
            channels = [24, 116, 232, 464]
        elif width_mult == 1.5:
            channels = [24, 176, 352, 704]
        elif width_mult == 2.0:
            channels = [24, 244, 488, 976]
        else:
            raise ValueError(f"Unsupported width multiplier: {width_mult}")
        
        self.stage_repeats = [4, 8, 4]
        
        # First conv (stride=1 for CIFAR!)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Build stages
        self.stage2 = self._make_stage(channels[0], channels[1], self.stage_repeats[0], first_stride=2)  # 32->16
        self.stage3 = self._make_stage(channels[1], channels[2], self.stage_repeats[1], first_stride=2)  # 16->8
        self.stage4 = self._make_stage(channels[2], channels[3], self.stage_repeats[2], first_stride=2)  # 8->4
        
        self.features = nn.Sequential(
            self.conv1,
            self.stage2,
            self.stage3,
            self.stage4
        )
        
        # Final conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Linear(1024, num_classes)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride):
        """Create a stage with multiple shuffle blocks."""
        layers = []
        layers.append(ShuffleBlock(in_channels, out_channels, stride=first_stride))
        for _ in range(num_blocks - 1):
            layers.append(ShuffleBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ShuffleBlock(nn.Module):
    """Basic ShuffleNetV2 block."""
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        
        if stride == 1:
            # Split channels
            self.branch_channels = in_channels // 2
            assert self.branch_channels * 2 == in_channels
        else:
            # Don't split when downsampling
            self.branch_channels = in_channels
        
        out_channels = out_channels - self.branch_channels
        
        # Branch 1 (when stride=1, this is identity; when stride=2, this is downsampling)
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.branch_channels, self.branch_channels, 3, stride, 1, 
                         groups=self.branch_channels, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.Conv2d(self.branch_channels, self.branch_channels, 1, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.ReLU(inplace=True)
            )
        
        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.branch_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        
        # Channel shuffle
        return self._channel_shuffle(out, 2)
    
    def _channel_shuffle(self, x, groups):
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        
        # Reshape
        x = x.view(batch, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, height, width)
        
        return x


def mobilenetv3_cifar(num_classes=10, width_mult=1.0):
    """MobileNetV3-Small for CIFAR-10/100."""
    return MobileNetV3_CIFAR(num_classes=num_classes, width_mult=width_mult)


def shufflenetv2_cifar(num_classes=10, width_mult=1.0):
    """ShuffleNetV2 for CIFAR-10/100."""
    return ShuffleNetV2_CIFAR(num_classes=num_classes, width_mult=width_mult)


if __name__ == "__main__":
    # Test models
    for name, model_fn, width in [
        ('MobileNetV3 (1.0x)', mobilenetv3_cifar, 1.0),
        ('MobileNetV3 (0.75x)', mobilenetv3_cifar, 0.75),
        ('ShuffleNetV2 (1.0x)', shufflenetv2_cifar, 1.0),
        ('ShuffleNetV2 (0.5x)', shufflenetv2_cifar, 0.5),
    ]:
        model = model_fn(num_classes=10, width_mult=width)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:25s}: {num_params:>8,} params, output: {y.shape}")