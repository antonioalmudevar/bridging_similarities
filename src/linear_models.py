"""
Linear Models for Representation Learning

This module provides linear (fully-connected) models for comparing with
convolutional models. These models flatten input images and use only
linear layers, making representation comparison straightforward without
needing spatial aggregation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class LinearNet(nn.Module):
    """
    Simple multi-layer linear network for image classification.
    
    Architecture:
        Input (flattened) -> Linear -> ReLU -> ... -> Linear -> ReLU -> Output
    
    Args:
        input_size: Size of flattened input (e.g., 3*32*32 = 3072 for CIFAR-10)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability (0 means no dropout)
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 num_classes: int,
                 use_batchnorm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Store layer boundaries for easy extraction
        self._compute_layer_boundaries()
    
    def _compute_layer_boundaries(self):
        """Compute indices where each 'layer' ends (for stitching)."""
        self.layer_boundaries = []
        
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                # Each Linear layer is a boundary point
                self.layer_boundaries.append(i)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten input: (B, C, H, W) -> (B, C*H*W)
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_layer_output(self, x, layer_idx: int):
        """
        Get output after a specific layer.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_idx: Index of the layer (0-indexed, corresponds to Linear layers)
        
        Returns:
            Output tensor after the specified layer (B, features)
        """
        x = x.view(x.size(0), -1)
        
        # Get the module index corresponding to this layer
        if layer_idx >= len(self.layer_boundaries):
            raise ValueError(f"Layer index {layer_idx} out of range (max: {len(self.layer_boundaries)-1})")
        
        module_idx = self.layer_boundaries[layer_idx]
        
        # Forward through network up to (and including) this layer
        for i, module in enumerate(self.network):
            x = module(x)
            if i == module_idx:
                break
        
        return x
    
    def get_num_layers(self):
        """Get number of Linear layers in the network."""
        return len(self.layer_boundaries)


def create_linear_model(model_name: str, 
                       input_size: int,
                       num_classes: int) -> LinearNet:
    """
    Factory function to create predefined linear model architectures.
    
    Args:
        model_name: Name of the architecture ('linear_small', 'linear_medium', 'linear_large')
        input_size: Size of flattened input
        num_classes: Number of output classes
    
    Returns:
        LinearNet model
    """
    
    if model_name == 'linear_small':
        # Small: 3 layers, ~100K parameters
        hidden_sizes = [512, 256]
        
    elif model_name == 'linear_medium':
        # Medium: 4 layers, ~500K parameters
        hidden_sizes = [1024, 512, 256]
        
    elif model_name == 'linear_large':
        # Large: 5 layers, ~1M parameters
        hidden_sizes = [2048, 1024, 512, 256]
        
    elif model_name == 'linear_deep':
        # Deep: 6 layers, similar total params but deeper
        hidden_sizes = [512, 512, 256, 256, 128]
    
    elif model_name == 'linear_wide':
        # Wide: 3 layers but very wide
        hidden_sizes = [2048, 1024]
    
    else:
        raise ValueError(f"Unknown linear model: {model_name}. "
                        f"Choose from: linear_small, linear_medium, linear_large, "
                        f"linear_deep, linear_wide")
    
    return LinearNet(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        use_batchnorm=True,
        dropout=0.1
    )


def get_linear_model_info(model_name: str) -> dict:
    """Get information about a linear model architecture."""
    
    info = {
        'linear_small': {
            'hidden_sizes': [512, 256],
            'description': 'Small linear network (3 layers, ~100K params)',
            'num_layers': 3  # 2 hidden + 1 output
        },
        'linear_medium': {
            'hidden_sizes': [1024, 512, 256],
            'description': 'Medium linear network (4 layers, ~500K params)',
            'num_layers': 4
        },
        'linear_large': {
            'hidden_sizes': [2048, 1024, 512, 256],
            'description': 'Large linear network (5 layers, ~1M params)',
            'num_layers': 5
        },
        'linear_deep': {
            'hidden_sizes': [512, 512, 256, 256, 128],
            'description': 'Deep linear network (6 layers)',
            'num_layers': 6
        },
        'linear_wide': {
            'hidden_sizes': [2048, 1024],
            'description': 'Wide linear network (3 layers, wide hidden)',
            'num_layers': 3
        }
    }
    
    return info.get(model_name, None)


def print_model_summary(model: LinearNet):
    """Print a summary of the linear model architecture."""
    
    print(f"\nLinear Model Summary:")
    print(f"  Input size: {model.input_size}")
    print(f"  Hidden layers: {model.hidden_sizes}")
    print(f"  Output classes: {model.num_classes}")
    print(f"  Total Linear layers: {model.get_num_layers()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\n  Layer structure:")
    
    for i, (name, module) in enumerate(model.network.named_children()):
        if isinstance(module, nn.Linear):
            print(f"    Layer {i}: {module.in_features} -> {module.out_features}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Linear Models\n")
    print("=" * 60)
    
    # Create example model
    model = create_linear_model('linear_medium', input_size=3072, num_classes=10)
    print_model_summary(model)
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass:")
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)  # CIFAR-10 sized input
    
    print(f"  Input shape: {x.shape}")
    output = model(x)
    print(f"  Output shape: {output.shape}")
    
    # Test layer extraction
    print("\n" + "=" * 60)
    print("Testing layer output extraction:")
    for layer_idx in range(model.get_num_layers()):
        layer_output = model.get_layer_output(x, layer_idx)
        print(f"  Layer {layer_idx} output shape: {layer_output.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
