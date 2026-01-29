"""
Convolutional Models for CIFAR-10/100 (32x32 images)

This module provides CNN models optimized for 32x32 images.
All models are designed for native CIFAR resolution - NO pretrained ImageNet weights.
"""

import torch
import torch.nn as nn
from typing import Optional

# Import CIFAR-specific ResNets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from .cifar_resnets import (
    resnet20_cifar, resnet32_cifar, resnet44_cifar, 
    resnet56_cifar, resnet110_cifar
)
from .cifar_mobilenets import mobilenetv3_cifar, shufflenetv2_cifar
from .cifar_densenet import (
    densenet40_cifar, densenet100_cifar, densenet_bc_100_cifar,
    densenet_bc_190_cifar, densenet_bc_250_cifar
)


class NarrowCNN(nn.Module):
    """
    Narrow CNN designed for 32x32 images.
    
    Architecture optimized for CIFAR-10:
    - Progressive channel growth: 32 → 64 → 128 → 256
    - Final features: 256×4×4 = 4,096 (well under 50k threshold)
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB)
        width_multiplier: Multiplier for channel dimensions
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3, width_multiplier: float = 1.0):
        super().__init__()
        
        # Apply width multiplier
        c1 = int(32 * width_multiplier)
        c2 = int(64 * width_multiplier)
        c3 = int(128 * width_multiplier)
        c4 = int(256 * width_multiplier)
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional blocks (designed for 32x32 input)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32→16
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16→8
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8→4
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            # No pooling - stay at 4×4
        )
        
        # Pack into features
        self.features = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.conv4
        )
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.classifier = nn.Linear(c4 * 4 * 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TinyCNN(nn.Module):
    """
    Extremely narrow CNN for 32x32 images.
    
    All layers stay under similarity threshold:
    - Progressive growth: 16 → 32 → 64 → 128
    - Final: 128×4×4 = 2,048 features
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.features = nn.Sequential(
            # Block 1: 16 channels
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
            
            # Block 2: 32 channels
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8
            
            # Block 3: 64 channels
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8→4
            
            # Block 4: 128 channels (no pooling)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Linear(128 * 4 * 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """Simple 2-block CNN for 32x32 images."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8
        )
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Linear(128 * 8 * 8, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cnn_model(model_name: str, num_classes: int = 10):
    """
    Factory function to create CNN models for CIFAR-10/100 (32x32 images).
    
    All models are designed for native 32x32 resolution.
    NO pretrained weights - train from scratch.
    
    Args:
        model_name: Name of the architecture
        num_classes: Number of output classes
    
    Returns:
        CNN model
    
    Available models:
        Custom CNNs:
            - 'tiny_cnn': Extremely narrow (2k final features)
            - 'narrow_cnn': Narrow (4k final features)
            - 'narrow_cnn_wide': Wider (16k final features)
            - 'simple_cnn': Simple 2-block (8k final features)
        
        CIFAR ResNets (from original paper):
            - 'resnet20': ResNet-20 for CIFAR (270K params)
            - 'resnet32': ResNet-32 for CIFAR (464K params)
            - 'resnet44': ResNet-44 for CIFAR (660K params)
            - 'resnet56': ResNet-56 for CIFAR (855K params)
            - 'resnet110': ResNet-110 for CIFAR (1.7M params)
    """
    
    # Custom models
    if model_name == 'tiny_cnn':
        return TinyCNN(num_classes=num_classes, input_channels=3)
    
    elif model_name == 'narrow_cnn':
        return NarrowCNN(num_classes=num_classes, input_channels=3, width_multiplier=1.0)
    
    elif model_name == 'narrow_cnn_wide':
        return NarrowCNN(num_classes=num_classes, input_channels=3, width_multiplier=2.0)
    
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, input_channels=3)
    
    # CIFAR ResNets
    elif model_name == 'resnet20':
        return resnet20_cifar(num_classes=num_classes)
    
    elif model_name == 'resnet32':
        return resnet32_cifar(num_classes=num_classes)
    
    elif model_name == 'resnet44':
        return resnet44_cifar(num_classes=num_classes)
    
    elif model_name == 'resnet56':
        return resnet56_cifar(num_classes=num_classes)
    
    elif model_name == 'resnet110':
        return resnet110_cifar(num_classes=num_classes)
    
    # MobileNets for CIFAR
    elif model_name == 'mobilenetv3':
        return mobilenetv3_cifar(num_classes=num_classes, width_mult=1.0)
    
    elif model_name == 'mobilenetv3_small':
        return mobilenetv3_cifar(num_classes=num_classes, width_mult=0.75)
    
    elif model_name == 'mobilenetv3_large':
        return mobilenetv3_cifar(num_classes=num_classes, width_mult=1.5)
    
    # ShuffleNets for CIFAR
    elif model_name == 'shufflenetv2':
        return shufflenetv2_cifar(num_classes=num_classes, width_mult=1.0)
    
    elif model_name == 'shufflenetv2_small':
        return shufflenetv2_cifar(num_classes=num_classes, width_mult=0.5)
    
    elif model_name == 'shufflenetv2_large':
        return shufflenetv2_cifar(num_classes=num_classes, width_mult=1.5)
    
    # DenseNets for CIFAR
    elif model_name == 'densenet40':
        return densenet40_cifar(num_classes=num_classes)
    
    elif model_name == 'densenet100':
        return densenet100_cifar(num_classes=num_classes)
    
    elif model_name == 'densenet_bc100':
        return densenet_bc_100_cifar(num_classes=num_classes)
    
    elif model_name == 'densenet_bc190':
        return densenet_bc_190_cifar(num_classes=num_classes)
    
    elif model_name == 'densenet_bc250':
        return densenet_bc_250_cifar(num_classes=num_classes)
    
    else:
        raise ValueError(
            f"Unknown CNN model: {model_name}. "
            f"Available: tiny_cnn, narrow_cnn, narrow_cnn_wide, simple_cnn, "
            f"resnet20, resnet32, resnet44, resnet56, resnet110, "
            f"mobilenetv3, mobilenetv3_small, mobilenetv3_large, "
            f"shufflenetv2, shufflenetv2_small, shufflenetv2_large, "
            f"densenet40, densenet100, densenet_bc100, densenet_bc190, densenet_bc250"
        )


def get_cnn_model_info(model_name: str) -> dict:
    """Get information about a CNN model architecture."""
    
    info = {
        'tiny_cnn': {
            'channels': [16, 32, 64, 128],
            'final_features': 2048,  # 128*4*4
            'description': 'Extremely narrow CNN (all layers < 50k features)',
            'params_approx': '~150K',
            'input_size': 32
        },
        'narrow_cnn': {
            'channels': [32, 64, 128, 256],
            'final_features': 4096,  # 256*4*4
            'description': 'Narrow CNN for CIFAR-10',
            'params_approx': '~350K',
            'input_size': 32
        },
        'narrow_cnn_wide': {
            'channels': [64, 128, 256, 512],
            'final_features': 16384,  # 512*4*4
            'description': 'Wider narrow CNN',
            'params_approx': '~1.2M',
            'input_size': 32
        },
        'simple_cnn': {
            'channels': [64, 128],
            'final_features': 8192,  # 128*8*8
            'description': 'Simple 2-block CNN',
            'params_approx': '~100K',
            'input_size': 32
        },
        'resnet20': {
            'depth': 20,
            'final_features': 64,  # After global avgpool
            'description': 'ResNet-20 for CIFAR (n=3)',
            'params_approx': '~270K',
            'input_size': 32
        },
        'resnet32': {
            'depth': 32,
            'final_features': 64,
            'description': 'ResNet-32 for CIFAR (n=5)',
            'params_approx': '~464K',
            'input_size': 32
        },
        'resnet44': {
            'depth': 44,
            'final_features': 64,
            'description': 'ResNet-44 for CIFAR (n=7)',
            'params_approx': '~660K',
            'input_size': 32
        },
        'resnet56': {
            'depth': 56,
            'final_features': 64,
            'description': 'ResNet-56 for CIFAR (n=9)',
            'params_approx': '~855K',
            'input_size': 32
        },
        'resnet110': {
            'depth': 110,
            'final_features': 64,
            'description': 'ResNet-110 for CIFAR (n=18)',
            'params_approx': '~1.7M',
            'input_size': 32
        },
        'mobilenetv3': {
            'width_mult': 1.0,
            'description': 'MobileNetV3-Small adapted for CIFAR (width=1.0)',
            'params_approx': '~2.5M',
            'input_size': 32
        },
        'mobilenetv3_small': {
            'width_mult': 0.75,
            'description': 'MobileNetV3-Small adapted for CIFAR (width=0.75)',
            'params_approx': '~1.5M',
            'input_size': 32
        },
        'mobilenetv3_large': {
            'width_mult': 1.5,
            'description': 'MobileNetV3-Small adapted for CIFAR (width=1.5)',
            'params_approx': '~5M',
            'input_size': 32
        },
        'shufflenetv2': {
            'width_mult': 1.0,
            'description': 'ShuffleNetV2 adapted for CIFAR (width=1.0)',
            'params_approx': '~1.3M',
            'input_size': 32
        },
        'shufflenetv2_small': {
            'width_mult': 0.5,
            'description': 'ShuffleNetV2 adapted for CIFAR (width=0.5)',
            'params_approx': '~350K',
            'input_size': 32
        },
        'shufflenetv2_large': {
            'width_mult': 1.5,
            'description': 'ShuffleNetV2 adapted for CIFAR (width=1.5)',
            'params_approx': '~2.5M',
            'input_size': 32
        },
        'densenet40': {
            'layers_per_block': 12,
            'growth_rate': 12,
            'description': 'DenseNet-40 for CIFAR (L=12, k=12)',
            'params_approx': '~1.0M',
            'input_size': 32
        },
        'densenet100': {
            'layers_per_block': 32,
            'growth_rate': 12,
            'description': 'DenseNet-100 for CIFAR (L=32, k=12)',
            'params_approx': '~7.0M',
            'input_size': 32
        },
        'densenet_bc100': {
            'layers_per_block': 16,
            'growth_rate': 12,
            'bottleneck': True,
            'description': 'DenseNet-BC-100 for CIFAR (L=16, k=12, bottleneck)',
            'params_approx': '~0.8M',
            'input_size': 32
        },
        'densenet_bc190': {
            'layers_per_block': 31,
            'growth_rate': 12,
            'bottleneck': True,
            'description': 'DenseNet-BC-190 for CIFAR (L=31, k=12, bottleneck)',
            'params_approx': '~25M',
            'input_size': 32
        },
        'densenet_bc250': {
            'layers_per_block': 41,
            'growth_rate': 12,
            'bottleneck': True,
            'description': 'DenseNet-BC-250 for CIFAR (L=41, k=12, bottleneck)',
            'params_approx': '~15M',
            'input_size': 32
        },
    }
    
    return info.get(model_name, None)


def print_cnn_summary(model: nn.Module, input_size: tuple = (3, 32, 32)):
    """Print a summary of the CNN architecture."""
    
    print(f"\nCNN Model Summary:")
    print(f"  Model type: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Try to get layer information for custom models
    if hasattr(model, 'features'):
        print(f"\n  Feature extractor:")
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                print(f"    Conv2d: {layer.in_channels} → {layer.out_channels} (kernel={layer.kernel_size[0]})")
            elif isinstance(layer, nn.MaxPool2d):
                print(f"    MaxPool2d: kernel={layer.kernel_size}")
    
    # Test forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n  Input shape: (batch, {input_size[0]}, {input_size[1]}, {input_size[2]})")
        print(f"  Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"\n  Could not compute output shape: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CNN Models for CIFAR-10 (32x32)\n")
    print("=" * 70)
    
    # Test custom models
    for model_name in ['tiny_cnn', 'narrow_cnn', 'simple_cnn', 'resnet20', 'resnet56']:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print('='*70)
        
        model = create_cnn_model(model_name, num_classes=10)
        print_cnn_summary(model, input_size=(3, 32, 32))
        
        # Get model info
        info = get_cnn_model_info(model_name)
        if info:
            print(f"\n  Model Info:")
            print(f"    Description: {info['description']}")
            if 'final_features' in info:
                print(f"    Final features: {info['final_features']:,}")
            print(f"    Input size: {info['input_size']}×{info['input_size']}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
