import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from .similarity_metrics import RepresentationalSimilarity


class ConvStitcher(nn.Module):
    """1x1 Convolutional stitcher for connecting layers with different channel dimensions."""
    
    def __init__(self, in_channels: int, out_channels: int, spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.spatial_size = spatial_size
        
        # Initialize to identity for self-stitching (easier convergence)
        if in_channels == out_channels:
            # Identity initialization
            with torch.no_grad():
                self.conv.weight.copy_(torch.eye(in_channels, out_channels).unsqueeze(-1).unsqueeze(-1))
                self.conv.bias.zero_()
        else:
            # Kaiming initialization for different dimensions
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1x1 convolution and spatial adaptation if needed."""
        x = self.conv(x)
        
        # Handle spatial size mismatch if needed
        if self.spatial_size is not None and x.shape[2:] != self.spatial_size:
            x = nn.functional.adaptive_avg_pool2d(x, self.spatial_size)
        
        return x


class InvertibleAffineConvStitcher(nn.Module):
    """Invertible affine 1x1 convolutional stitcher.
    
    Learns: y = W * x + b where W is invertible.
    Requires in_channels == out_channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int, spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("InvertibleAffineConvStitcher requires in_channels == out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        
        # Parameterize invertible weight via matrix exponential: W = expm(A)
        self.log_weight = nn.Parameter(torch.zeros(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply invertible affine 1x1 convolution."""
        log_weight = self.log_weight.reshape(self.out_channels, self.in_channels)
        weight = torch.matrix_exp(log_weight).unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.conv2d(x, weight, bias=self.bias)
        
        if self.spatial_size is not None and x.shape[2:] != self.spatial_size:
            x = nn.functional.adaptive_avg_pool2d(x, self.spatial_size)
        
        return x


class OrthogonalConvStitcher(nn.Module):
    """Orthogonal 1x1 convolutional stitcher (no bias, orthogonal weight matrix).
    
    Learns: y = Q * x where Q is orthogonal (Q^T Q = I)
    This is a 1x1 convolution with orthogonal constraint on the weight matrix.
    
    The weight is shape (out_channels, in_channels, 1, 1), which we treat as
    a (out_channels, in_channels) matrix for orthogonality.
    
    Preserves:
    - Euclidean distances (for square Q)
    - Angles between feature vectors
    - Vector norms (for square Q)
    """
    
    def __init__(self, in_channels: int, out_channels: int, spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        
        # Initialize weight as (out_channels, in_channels, 1, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        # Initialize with identity for self-stitching, orthogonal otherwise
        with torch.no_grad():
            if in_channels == out_channels:
                # Identity initialization for same dimensions
                I = torch.eye(in_channels, out_channels)
                self.weight.copy_(I.unsqueeze(-1).unsqueeze(-1))
            else:
                # Orthogonal initialization for different dimensions
                # For rectangular matrices, initialize with semi-orthogonal matrix
                W = torch.randn(out_channels, in_channels)
                
                # Use SVD to get orthogonal initialization
                # For out < in: Q is (out, out), need to pad to (out, in)
                # For out > in: Q is (out, in) which is what we want
                if out_channels <= in_channels:
                    # More input than output: orthonormal rows
                    # Initialize first out_channels columns orthogonally, rest zeros
                    Q, _ = torch.linalg.qr(W.T)  # Q: (in, out)
                    W_init = Q.T  # (out, in)
                else:
                    # More output than input: orthonormal columns
                    Q, _ = torch.linalg.qr(W)  # Q: (out, in)
                    W_init = Q
                
                self.weight.copy_(W_init.unsqueeze(-1).unsqueeze(-1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonal 1x1 convolution."""
        # Use the learned weight directly (like LinearStitcher)
        # Orthogonality is enforced via regularization loss during training
        # NOT via QR projection (which breaks gradients)
        x = torch.nn.functional.conv2d(x, self.weight, bias=None)
        
        # Handle spatial size mismatch if needed
        if self.spatial_size is not None and x.shape[2:] != self.spatial_size:
            x = nn.functional.adaptive_avg_pool2d(x, self.spatial_size)
        
        return x
    
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality constraint loss for conv stitcher.
        
        For orthogonal matrix Q: Q^T Q = I
        Loss = ||Q^T Q - I||_F^2
        """
        W = self.weight.squeeze(-1).squeeze(-1)  # (out_channels, in_channels)
        WtW = torch.matmul(W.t(), W)  # (in_channels, in_channels) for W^T W = I
        
        # Target is identity
        n = min(self.out_channels, self.in_channels)
        I = torch.eye(n, device=W.device, dtype=W.dtype)
        
        if self.out_channels == self.in_channels:
            # Square: W^T W should be I
            loss = torch.norm(WtW - I, p='fro')**2
        else:
            # Rectangular: enforce orthonormality on the relevant part
            loss = torch.norm(WtW[:n, :n] - I, p='fro')**2
        
        return loss


class OrthogonalScaledConvStitcher(nn.Module):
    """Orthogonal + isotropic scaling 1x1 convolutional stitcher.
    
    Learns: y = s * Q * x where Q is orthogonal and s is a scalar
    This is a 1x1 convolution with orthogonal weight + uniform scaling.
    
    Invariant to:
    - Rotations
    - Reflections  
    - Uniform scaling
    
    This is more flexible than pure orthogonal but more constrained than affine.
    """
    
    def __init__(self, in_channels: int, out_channels: int, spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        
        # Orthogonal weight matrix (out_channels, in_channels, 1, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        # Learnable scalar scale
        self.scale = nn.Parameter(torch.ones(1))
        
        # Initialize with identity for self-stitching, orthogonal otherwise
        with torch.no_grad():
            if in_channels == out_channels:
                # Identity initialization
                I = torch.eye(in_channels, out_channels)
                self.weight.copy_(I.unsqueeze(-1).unsqueeze(-1))
            else:
                # Orthogonal initialization for rectangular matrices
                W = torch.randn(out_channels, in_channels)
                
                # Use SVD to get orthogonal initialization
                if out_channels <= in_channels:
                    # More input than output: orthonormal rows
                    Q, _ = torch.linalg.qr(W.T)  # Q: (in, out)
                    W_init = Q.T  # (out, in)
                else:
                    # More output than input: orthonormal columns
                    Q, _ = torch.linalg.qr(W)  # Q: (out, in)
                    W_init = Q
                
                self.weight.copy_(W_init.unsqueeze(-1).unsqueeze(-1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled orthogonal 1x1 convolution."""
        # Use the learned weight directly (like LinearStitcher)
        # Orthogonality enforced via regularization, not QR projection
        # Scale is applied to the weight
        weight_scaled = self.scale * self.weight
        
        x = torch.nn.functional.conv2d(x, weight_scaled, bias=None)
        
        # Handle spatial size mismatch if needed
        if self.spatial_size is not None and x.shape[2:] != self.spatial_size:
            x = nn.functional.adaptive_avg_pool2d(x, self.spatial_size)
        
        return x
    
    def orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality constraint loss for scaled conv stitcher."""
        W = self.weight.squeeze(-1).squeeze(-1)  # (out_channels, in_channels)
        WtW = torch.matmul(W.t(), W)  # (in_channels, in_channels)
        
        n = min(self.out_channels, self.in_channels)
        I = torch.eye(n, device=W.device, dtype=W.dtype)
        
        if self.out_channels == self.in_channels:
            loss = torch.norm(WtW - I, p='fro')**2
        else:
            loss = torch.norm(WtW[:n, :n] - I, p='fro')**2
        
        return loss
    
    def get_scale(self) -> float:
        """Get the learned scale parameter."""
        return self.scale.item()


class LinearStitcher(nn.Module):
    """Affine stitcher for connecting fully-connected layers with different dimensions.
    
    Learns: y = Wx + b (unconstrained linear transformation)
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # Initialize
        if in_features == out_features:
            # For same dimensions (e.g., self-stitching), initialize as identity
            # This makes it easier to learn identity transformation
            nn.init.eye_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        else:
            # For different dimensions, use Kaiming initialization
            nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation."""
        return self.linear(x)


class InvertibleAffineStitcher(nn.Module):
    """Invertible affine stitcher for connecting fully-connected layers.
    
    Learns: y = Wx + b where W is invertible.
    Requires in_features == out_features.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features != out_features:
            raise ValueError("InvertibleAffineStitcher requires in_features == out_features")
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameterize invertible weight via matrix exponential: W = expm(A)
        self.log_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply invertible affine transformation."""
        log_weight = self.log_weight.reshape(self.out_features, self.in_features)
        weight = torch.matrix_exp(log_weight)
        return torch.matmul(x, weight.t()) + self.bias


class OrthogonalStitcher(nn.Module):
    """Orthogonal stitcher for connecting fully-connected layers.
    
    Learns: y = Qx (orthogonal transformation, no bias)
    Where Q^T Q = I (orthogonal matrix)
    
    This preserves:
    - Euclidean distances: ||Qx - Qy|| = ||x - y||
    - Angles between vectors
    - Vector norms: ||Qx|| = ||x||
    
    Useful for understanding if representations differ only by rotation.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # For orthogonal matrices, we need in_features == out_features
        # For rectangular case, use truncated orthogonal (QR decomposition approach)
        if in_features == out_features:
            # Square orthogonal matrix
            # Initialize with a random orthogonal matrix
            Q = torch.nn.init.orthogonal_(torch.empty(out_features, in_features))
            self.weight = nn.Parameter(Q)
        else:
            # Rectangular case: use a tall/wide matrix
            # We'll enforce orthogonality on the relevant subspace
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W) if out_features <= in_features else nn.init.xavier_uniform_(W)
            self.weight = nn.Parameter(W)
        
        # No bias for orthogonal transformation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonal transformation."""
        # Project weight matrix to orthogonal manifold using Cayley transform
        # or just use the parameterized weight (trained with orthogonality regularization)
        return torch.matmul(x, self.weight.t())
    
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality constraint loss.
        
        For orthogonal matrix Q: Q^T Q = I
        Loss = ||Q^T Q - I||_F^2
        """
        W = self.weight
        WtW = torch.matmul(W, W.t())
        
        # Target is identity for square, or I for the smaller dimension
        n = min(self.out_features, self.in_features)
        I = torch.eye(n, device=W.device, dtype=W.dtype)
        
        if self.out_features == self.in_features:
            # Square: Q^T Q should be I
            loss = torch.norm(WtW - I, p='fro')**2
        else:
            # Rectangular: enforce orthonormality on the relevant part
            loss = torch.norm(WtW[:n, :n] - I, p='fro')**2
        
        return loss


class OrthogonalScaledStitcher(nn.Module):
    """Orthogonal + isotropic scaling stitcher.
    
    Learns: y = s * Q * x (orthogonal transformation + uniform scale, no bias)
    Where:
    - Q^T Q = I (orthogonal matrix - rotation/reflection)
    - s is a positive scalar (uniform scale)
    
    This preserves:
    - Angles between vectors (up to scale)
    - Relative distances (up to scale)
    
    Invariant to:
    - Rotations, reflections, uniform scaling
    
    Useful for understanding if representations differ by rotation + scale.
    This is the transformation that CKA and Orth+Scale metrics are invariant to.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Orthogonal weight matrix (same as OrthogonalStitcher)
        if in_features == out_features:
            Q = torch.nn.init.orthogonal_(torch.empty(out_features, in_features))
            self.weight = nn.Parameter(Q)
        else:
            W = torch.empty(out_features, in_features)
            nn.init.orthogonal_(W) if out_features <= in_features else nn.init.xavier_uniform_(W)
            self.weight = nn.Parameter(W)
        
        # Learnable isotropic scale (initialized to 1.0)
        self.log_scale = nn.Parameter(torch.zeros(1))  # Use log for numerical stability
        
        # No bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonal transformation with uniform scaling."""
        # Apply orthogonal transformation
        rotated = torch.matmul(x, self.weight.t())
        # Apply isotropic scale (exp to ensure positive)
        scale = torch.exp(self.log_scale)
        return scale * rotated
    
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality constraint loss.
        Same as OrthogonalStitcher.
        """
        W = self.weight
        WtW = torch.matmul(W, W.t())
        
        n = min(self.out_features, self.in_features)
        I = torch.eye(n, device=W.device, dtype=W.dtype)
        
        if self.out_features == self.in_features:
            loss = torch.norm(WtW - I, p='fro')**2
        else:
            loss = torch.norm(WtW[:n, :n] - I, p='fro')**2
        
        return loss
    
    def get_scale(self) -> float:
        """Get the current scale value."""
        return torch.exp(self.log_scale).item()


class StitchedNetwork(nn.Module):
    """
    A stitched network that:
    1. Runs input through source model up to source_layer
    2. Applies stitcher transformation  
    3. Continues through target model from target_layer onwards
    
    This implementation manually propagates through target model layers to avoid
    batch normalization issues.
    """
    
    def __init__(self, source_model: nn.Module, target_model: nn.Module,
                 source_layer_name: str, target_layer_name: str,
                 stitcher: ConvStitcher):
        super().__init__()
        
        self.source_model = source_model
        self.target_model = target_model
        self.stitcher = stitcher
        self.source_layer_name = source_layer_name
        self.target_layer_name = target_layer_name
        
        # Storage for source activation
        self.source_activation = None
        self._register_source_hook()
        
        # Build the remaining target model path
        self._build_target_path()
    
    def _register_source_hook(self):
        """Register hook to capture source layer output."""
        source_layer = self._find_layer(self.source_model, self.source_layer_name)
        if source_layer is None:
            raise ValueError(f"Source layer {self.source_layer_name} not found")
        
        def capture_hook(module, input, output):
            self.source_activation = output
        
        source_layer.register_forward_hook(capture_hook)
    
    def _find_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find a layer by name in the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def _is_mobilenet_like(self) -> bool:
        """Heuristic check for MobileNet-style models."""
        model_name = self.target_model.__class__.__name__
        if 'MobileNet' in model_name:
            return True
        for module in self.target_model.modules():
            if module.__class__.__name__ == 'InvertedResidual':
                return True
        return False

    def _is_squeezenet_like(self) -> bool:
        """Heuristic check for SqueezeNet-style models."""
        model_name = self.target_model.__class__.__name__
        if 'SqueezeNet' in model_name:
            return True
        for module in self.target_model.modules():
            if module.__class__.__name__ == 'Fire':
                return True
        return False
    
    def _build_target_path(self):
        """Build the computational path from target_layer to output."""
        # For LinearNet models (check for 'network' attribute which is a Sequential)
        if hasattr(self.target_model, 'network') and isinstance(self.target_model.network, nn.Sequential):
            self.model_type = 'linear'
        # For ResNet models
        elif hasattr(self.target_model, 'layer1'):
            self.model_type = 'resnet'
        # For SqueezeNet models
        elif self._is_squeezenet_like():
            self.model_type = 'squeezenet'
            self.target_classifier_idx = None
            if self.target_layer_name.startswith('features.'):
                parts = self.target_layer_name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    self.target_layer_idx = int(parts[1])
                else:
                    self.model_type = 'unknown'
            elif self.target_layer_name.startswith('classifier.'):
                self.target_classifier_idx = int(self.target_layer_name.split('.')[-1])
            else:
                self.model_type = 'unknown'
        # For ShuffleNet (torchvision-style)
        elif (hasattr(self.target_model, 'stage2') and hasattr(self.target_model, 'stage3') and
              hasattr(self.target_model, 'stage4') and hasattr(self.target_model, 'conv5')):
            self.model_type = 'shufflenet'
        # For DenseNet models (custom: dense1/trans1 attributes)
        elif hasattr(self.target_model, 'dense1') and hasattr(self.target_model, 'trans1'):
            self.model_type = 'densenet'
        # For torchvision-style DenseNet (features Sequential with denseblock/transition)
        elif (hasattr(self.target_model, 'features') and hasattr(self.target_model, 'classifier') and
              isinstance(self.target_model.features, nn.Sequential) and
              any(name.startswith('denseblock') for name, _ in self.target_model.features.named_children())):
            self.model_type = 'densenet_features'
            layer_key = self.target_layer_name
            if layer_key.startswith('features.'):
                layer_key = layer_key.split('.', 1)[1]
            # Map nested names like "denseblock1.denselayer1" to top-level block
            layer_key = layer_key.split('.')[0]
            for idx, (name, _) in enumerate(self.target_model.features.named_children()):
                if name == layer_key:
                    self.target_layer_idx = idx
                    break
            else:
                self.model_type = 'unknown'
        # For custom CNNs (NarrowCNN, TinyCNN, SimpleCNN) - have Sequential features AND conv1/conv2 attributes
        elif (hasattr(self.target_model, 'features') and hasattr(self.target_model, 'classifier') and
              hasattr(self.target_model, 'conv1') and isinstance(self.target_model.conv1, nn.Sequential)):
            # ShuffleNet has stage2/3/4 + conv5 and needs special handling
            if (hasattr(self.target_model, 'stage2') and hasattr(self.target_model, 'stage3') and
                hasattr(self.target_model, 'stage4') and hasattr(self.target_model, 'conv5')):
                self.model_type = 'shufflenet'
            else:
                self.model_type = 'custom_cnn'
            
            # Handle layer names (custom CNN path only)
            if self.model_type == 'custom_cnn':
                if any(self.target_layer_name.startswith(f'conv{i}') for i in range(1, 10)):
                    # Layer like "conv1", "conv2", etc. (NarrowCNN block-level names)
                    conv_num = int(self.target_layer_name.replace('conv', ''))
                    self.target_layer_idx = conv_num - 1  # conv1 -> index 0, conv2 -> index 1, etc.
                elif 'features' in self.target_layer_name:
                    # Layer like "features.0"
                    self.target_layer_idx = int(self.target_layer_name.split('.')[-1])
                else:
                    self.model_type = 'unknown'
        # For SimpleCNN-like models: features + classifier, no avgpool, no conv blocks
        elif (hasattr(self.target_model, 'features') and hasattr(self.target_model, 'classifier') and
              not hasattr(self.target_model, 'avgpool') and
              (not hasattr(self.target_model, 'conv1') or not isinstance(self.target_model.conv1, nn.Sequential)) and
              not self._is_mobilenet_like() and not self._is_squeezenet_like()):
            self.model_type = 'custom_cnn'
            
            if 'features' in self.target_layer_name:
                self.target_layer_idx = int(self.target_layer_name.split('.')[-1])
            else:
                self.model_type = 'unknown'
        # For MobileNet/ShuffleNet (Sequential features but NO conv1 attribute)
        elif (hasattr(self.target_model, 'features') and hasattr(self.target_model, 'classifier') and
              isinstance(self.target_model.features, nn.Sequential) and
              self._is_mobilenet_like()):
            self.model_type = 'mobilenet'
            
            if 'features' in self.target_layer_name:
                self.target_layer_idx = int(self.target_layer_name.split('.')[-1])
            else:
                self.model_type = 'unknown'
        # For VGG models (must have avgpool!)
        elif hasattr(self.target_model, 'features') and hasattr(self.target_model, 'classifier') and hasattr(self.target_model, 'avgpool'):
            if 'features' in self.target_layer_name:
                self.model_type = 'vgg'
                self.target_layer_idx = int(self.target_layer_name.split('.')[-1])
            else:
                self.model_type = 'unknown'
        else:
            self.model_type = 'unknown'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stitched network."""
        
        # Step 1: Get features from source model
        self.source_activation = None
        with torch.no_grad():
            _ = self.source_model(x)
        
        if self.source_activation is None:
            raise RuntimeError("Source activation was not captured")
        
        # Step 2: Apply stitcher
        features = self.stitcher(self.source_activation)
        
        # Step 3: Continue through target model
        if self.model_type == 'vgg':
            # Forward through remaining VGG feature layers
            for idx in range(self.target_layer_idx + 1, len(self.target_model.features)):
                features = self.target_model.features[idx](features)
            
            # Through avgpool and classifier
            features = self.target_model.avgpool(features)
            features = torch.flatten(features, 1)
            output = self.target_model.classifier(features)
        
        elif self.model_type == 'custom_cnn':
            # For custom CNNs (TinyCNN, NarrowCNN, SimpleCNN)
            # Forward through remaining feature layers
            for idx in range(self.target_layer_idx + 1, len(self.target_model.features)):
                features = self.target_model.features[idx](features)
            
            # Apply adaptive pooling if present (handles non-32x32 inputs)
            if hasattr(self.target_model, 'pool'):
                features = self.target_model.pool(features)

            # Flatten and through classifier (no avgpool in custom CNNs)
            features = torch.flatten(features, 1)
            output = self.target_model.classifier(features)
        
        elif self.model_type == 'squeezenet':
            # For SqueezeNet: features -> classifier (conv + relu + avgpool)
            if self.target_layer_name.startswith('features.'):
                for idx in range(self.target_layer_idx + 1, len(self.target_model.features)):
                    features = self.target_model.features[idx](features)
                output = self.target_model.classifier(features)
            elif self.target_layer_name.startswith('classifier.'):
                for idx in range(self.target_classifier_idx + 1, len(self.target_model.classifier)):
                    features = self.target_model.classifier[idx](features)
                output = features
            else:
                raise RuntimeError(f"Unknown SqueezeNet layer name: {self.target_layer_name}")
            
            # SqueezeNet classifier returns [N,C,1,1]; flatten for CE loss
            if output.dim() > 2:
                output = torch.flatten(output, 1)
            
        elif self.model_type == 'resnet':
            # For ResNet, we need to continue through remaining computation
            # Strategy: Run a modified forward pass that starts from the stitched features
            
            # We need to figure out which layer we stitched and continue from there
            # For CIFAR ResNet: conv1 -> bn1 -> relu -> layer1 -> layer2 -> layer3 -> avgpool -> fc
            
            # Determine continuation point based on target layer name
            has_layer4 = hasattr(self.target_model, 'layer4')
            if self.target_layer_name == 'conv1':
                # After conv1, still need: bn1, relu, then layer1, layer2, layer3 (+ layer4 for ImageNet ResNet)
                features = self.target_model.bn1(features)
                features = torch.relu(features)
                features = self.target_model.layer1(features)
                features = self.target_model.layer2(features)
                features = self.target_model.layer3(features)
                if has_layer4:
                    features = self.target_model.layer4(features)
            elif self.target_layer_name.startswith('layer1'):
                # Continue from within layer1 block, then layer2 and layer3
                parts = self.target_layer_name.split('.')
                block_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if block_idx is not None:
                    for i in range(block_idx + 1, len(self.target_model.layer1)):
                        features = self.target_model.layer1[i](features)
                features = self.target_model.layer2(features)
                features = self.target_model.layer3(features)
                if has_layer4:
                    features = self.target_model.layer4(features)
            elif self.target_layer_name.startswith('layer2'):
                # Continue from within layer2 block, then layer3
                parts = self.target_layer_name.split('.')
                block_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if block_idx is not None:
                    for i in range(block_idx + 1, len(self.target_model.layer2)):
                        features = self.target_model.layer2[i](features)
                features = self.target_model.layer3(features)
                if has_layer4:
                    features = self.target_model.layer4(features)
            elif self.target_layer_name.startswith('layer3'):
                # Continue from within layer3 block
                parts = self.target_layer_name.split('.')
                block_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if block_idx is not None:
                    for i in range(block_idx + 1, len(self.target_model.layer3)):
                        features = self.target_model.layer3[i](features)
                if has_layer4:
                    features = self.target_model.layer4(features)
            elif self.target_layer_name.startswith('layer4') and has_layer4:
                # Continue from within layer4 block (ImageNet ResNet)
                parts = self.target_layer_name.split('.')
                block_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if block_idx is not None:
                    for i in range(block_idx + 1, len(self.target_model.layer4)):
                        features = self.target_model.layer4[i](features)
            
            # Final pooling and classifier
            features = self.target_model.avgpool(features)
            features = torch.flatten(features, 1)
            output = self.target_model.fc(features)
            
        elif self.model_type == 'linear':
            # Linear model - features are already 2D, just continue through remaining layers
            # Find the target layer index in the network
            target_layer = self._find_layer(self.target_model, self.target_layer_name)
            if target_layer is None:
                raise RuntimeError(f"Target layer {self.target_layer_name} not found")
            
            # Find the index of the target layer in the network.Sequential
            target_idx = None
            for i, (name, module) in enumerate(self.target_model.network.named_children()):
                if name == self.target_layer_name.split('.')[-1]:
                    target_idx = i
                    break
            
            if target_idx is None:
                raise RuntimeError(f"Could not find target layer index for {self.target_layer_name}")
            
            # Forward through remaining layers
            for i in range(target_idx + 1, len(self.target_model.network)):
                features = self.target_model.network[i](features)
            
            output = features
        
        elif self.model_type == 'densenet':
            # For DenseNet, continue through remaining dense blocks + transitions
            # Layer names: conv1, dense1, trans1, dense2, trans2, dense3, bn, fc
            apply_bn = True
            
            if self.target_layer_name == 'conv1':
                # After conv1, continue through all dense blocks
                features = self.target_model.dense1(features)
                features = self.target_model.trans1(features)
                features = self.target_model.dense2(features)
                features = self.target_model.trans2(features)
                features = self.target_model.dense3(features)
            elif self.target_layer_name == 'dense1':
                # After dense1, apply transition then remaining blocks
                features = self.target_model.trans1(features)
                features = self.target_model.dense2(features)
                features = self.target_model.trans2(features)
                features = self.target_model.dense3(features)
            elif self.target_layer_name == 'trans1':
                # After trans1, continue with dense2
                features = self.target_model.dense2(features)
                features = self.target_model.trans2(features)
                features = self.target_model.dense3(features)
            elif self.target_layer_name == 'dense2':
                # After dense2, apply transition then dense3
                features = self.target_model.trans2(features)
                features = self.target_model.dense3(features)
            elif self.target_layer_name == 'trans2':
                # After trans2, continue with dense3
                features = self.target_model.dense3(features)
            elif self.target_layer_name == 'dense3':
                # Already at dense3 output
                pass
            elif self.target_layer_name == 'bn':
                # Already at BN output; skip BN below
                apply_bn = False
            else:
                raise RuntimeError(f"Unknown DenseNet layer name: {self.target_layer_name}")
            
            # Final processing
            if apply_bn:
                features = F.relu(self.target_model.bn(features))
            else:
                features = F.relu(features)
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
            output = self.target_model.fc(features)
        
        elif self.model_type == 'densenet_features':
            # For torchvision DenseNet, continue through remaining feature layers
            for idx in range(self.target_layer_idx + 1, len(self.target_model.features)):
                features = self.target_model.features[idx](features)
            
            # Final processing matches torchvision DenseNet forward
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, 1)
            features = torch.flatten(features, 1)
            output = self.target_model.classifier(features)
        
        elif self.model_type == 'mobilenet':
            # For MobileNet/ShuffleNet with Sequential features
            # Continue through remaining feature layers
            for idx in range(self.target_layer_idx + 1, len(self.target_model.features)):
                features = self.target_model.features[idx](features)
            
            # Apply final pooling and classifier
            # Check if model has explicit avgpool or conv5
            if hasattr(self.target_model, 'avgpool'):
                features = self.target_model.avgpool(features)
            elif hasattr(self.target_model, 'conv5'):
                # ShuffleNetV2 has conv5
                features = self.target_model.conv5(features)
                features = F.adaptive_avg_pool2d(features, 1)
            else:
                # Generic: just do adaptive avgpool
                features = F.adaptive_avg_pool2d(features, 1)
            
            features = features.view(features.size(0), -1)
            output = self.target_model.classifier(features)
        
        elif self.model_type == 'shufflenet':
            # For ShuffleNetV2 CIFAR: conv1 -> stage2 -> stage3 -> stage4 -> conv5 -> avgpool -> classifier
            if self.target_layer_name == 'conv1':
                features = self.target_model.stage2(features)
                features = self.target_model.stage3(features)
                features = self.target_model.stage4(features)
                features = self.target_model.conv5(features)
            elif self.target_layer_name == 'stage2':
                features = self.target_model.stage3(features)
                features = self.target_model.stage4(features)
                features = self.target_model.conv5(features)
            elif self.target_layer_name == 'stage3':
                features = self.target_model.stage4(features)
                features = self.target_model.conv5(features)
            elif self.target_layer_name == 'stage4':
                features = self.target_model.conv5(features)
            elif self.target_layer_name == 'conv5':
                pass
            else:
                raise RuntimeError(f"Unknown ShuffleNet layer name: {self.target_layer_name}")
            
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
            if hasattr(self.target_model, 'classifier'):
                output = self.target_model.classifier(features)
            elif hasattr(self.target_model, 'fc'):
                output = self.target_model.fc(features)
            else:
                # Fallback: use last linear if present
                last_linear = None
                for module in self.target_model.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                if last_linear is None:
                    raise RuntimeError("ShuffleNet head not found (no classifier/fc/linear layer)")
                output = last_linear(features)
            
        else:
            # Unknown model type - try adaptive pooling + last linear layer
            # Check if features are 2D or 4D
            if len(features.shape) == 2:
                # Already 2D (linear layer output)
                # Find the last linear layer
                last_linear = None
                for module in self.target_model.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                
                if last_linear is not None and features.shape[1] == last_linear.in_features:
                    output = last_linear(features)
                else:
                    raise RuntimeError(f"Cannot handle model - feature dim {features.shape[1]} doesn't match last linear layer")
            else:
                # 4D convolutional features
                features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                
                # Find the last linear layer
                last_linear = None
                for module in self.target_model.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                
                if last_linear is not None and features.shape[1] == last_linear.in_features:
                    output = last_linear(features)
                else:
                    # Can't proceed - raise error
                    raise RuntimeError(f"Cannot handle model type {type(self.target_model)}")
        
        return output


class ImprovedModelStitcher:
    """
    Improved model stitcher that properly optimizes cross-entropy loss.
    Supports affine, orthogonal, and orthogonal_scaled stitchers.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 similarity_aggregation: str = 'flatten',
                 include_intermediate: bool = False,
                 stitcher_type: str = 'affine',
                 max_features_for_similarity: int = 2048,
                 max_features_for_rank: int = 2048,
                 train_loss: str = 'ce',
                 kl_temperature: float = 1.0,
                 use_block_outputs: bool = True,
                 use_amp: bool = False):
        """
        Args:
            device: Device to run on ('cuda' or 'cpu')
            similarity_aggregation: How to aggregate spatial dimensions for similarity metrics
            include_intermediate: For linear models, whether to include intermediate layers
                                (BatchNorm, ReLU, Dropout) in addition to Linear layers.
                                Default False = only Linear layers (cleaner, fewer comparisons).
                                Set True = all layers (denser analysis, more comparisons).
            stitcher_type: Type of stitcher to use:
                          'affine' - Affine transformation (Wx + b), unconstrained
                          'orthogonal' - Orthogonal transformation (Qx), preserves distances
            max_features_for_similarity: Maximum number of features for slow similarity metrics.
                                        Fast metrics (CKA, SVCCA): always computed
                                        Slow metrics (RSA, L2, Procrustes, Orth+Scale): only if features <= this value.
            max_features_for_rank: Maximum number of features used when estimating representation rank.
                                  If feature_dim exceeds this, a random projection is used for a cheap approximation.
            train_loss: Training loss for the stitcher ('ce' or 'kl').
            kl_temperature: Temperature for KL loss (only used when train_loss='kl').
            use_block_outputs: If True (default), only use block output layers.
                              Reduces comparisons dramatically for faster computation.
                              Set to False to use all layers.
            use_amp: If True, use automatic mixed precision during stitcher training (CUDA only).
        """
        self.device = device
        self.similarity_aggregation = similarity_aggregation
        self.include_intermediate = include_intermediate
        self.stitcher_type = stitcher_type
        self.max_features_for_similarity = max_features_for_similarity
        self.max_features_for_rank = max_features_for_rank
        self.train_loss = train_loss
        self.kl_temperature = kl_temperature
        self.use_block_outputs = use_block_outputs
        self.use_amp = use_amp
        
        if stitcher_type not in ['affine', 'orthogonal', 'orthogonal_scaled']:
            raise ValueError(
                "stitcher_type must be 'affine', 'orthogonal', or 'orthogonal_scaled', "
                f"got '{stitcher_type}'"
            )
        if train_loss not in ['ce', 'kl']:
            raise ValueError(f"train_loss must be 'ce' or 'kl', got '{train_loss}'")
        if kl_temperature <= 0:
            raise ValueError(f"kl_temperature must be > 0, got '{kl_temperature}'")
    
    def filter_block_outputs(self, all_layers: List[str]) -> List[str]:
        """
        Filter layers to keep only block outputs for faster computation.
        
        Automatically detects architecture type and keeps:
        - ResNet: stem + block outputs (skip shortcuts)
        - MobileNet: stem + block outputs (skip SE modules) + head
        - ShuffleNet: stem + stage outputs + head
        - DenseNet: stem + denselayer outputs + transitions + head
        
        Args:
            all_layers: List of all layer names
        Returns:
            Filtered list of layer names
        """
        if not self.use_block_outputs:
            return all_layers
        
        # Detect architecture type from layer names
        arch_type = ''
        
        # Try to infer from layer names
        if any('layer1' in l for l in all_layers):
            arch_type = 'resnet'
        elif any('denseblock' in l for l in all_layers):
            arch_type = 'densenet'
        elif any('.squeeze' in l or '.expand1x1' in l or '.expand3x3' in l for l in all_layers):
            arch_type = 'squeezenet'
        elif any('features.' in l and '.conv.' in l for l in all_layers):
            arch_type = 'mobilenet'
        elif any('stage' in l for l in all_layers):
            arch_type = 'shufflenet'
        
        if not arch_type:
            # Could not detect - return all layers
            return all_layers
        
        relevant = []
        
        if 'resnet' in arch_type:
            # ResNet: Keep conv1 and block outputs (layerX.Y)
            if 'conv1' in all_layers:
                relevant.append('conv1')
            for layer in all_layers:
                if ('.conv2' in layer or '.conv3' in layer) and 'shortcut' not in layer:
                    # Map conv to its parent block output (after skip + relu)
                    block_name = layer.rsplit('.', 1)[0]
                    relevant.append(block_name)
        
        elif 'mobilenet' in arch_type:
            # MobileNet: Keep stem, block-level outputs, head
            # For MobileNetV3, blocks are features.0, features.1, features.2, etc.
            # We should NOT go inside blocks (features.3.conv.4) as they have complex residual structures
            
            # Collect all top-level feature blocks (features.N where N is a number)
            block_indices = set()
            for layer in all_layers:
                if layer.startswith('features.'):
                    parts = layer.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_idx = int(parts[1])
                        block_indices.add(block_idx)
            
            # Keep only the top-level blocks (not their internal layers)
            for idx in sorted(block_indices):
                block_name = f'features.{idx}'
                # Only add if it's a direct block (not an internal layer like features.3.conv.4)
                if block_name in all_layers:
                    relevant.append(block_name)
            
            # If we didn't find any top-level blocks, fall back to all layers
            # (this handles different MobileNet implementations)
            if len(relevant) == 0:
                return all_layers
                
        elif 'squeezenet' in arch_type:
            # SqueezeNet: Keep stem + Fire blocks + head
            if 'features.0' in all_layers:
                relevant.append('features.0')
            
            fire_indices = set()
            for layer in all_layers:
                if layer.startswith('features.') and ('.squeeze' in layer or '.expand' in layer):
                    parts = layer.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        fire_indices.add(int(parts[1]))
            
            for idx in sorted(fire_indices):
                relevant.append(f'features.{idx}')
            
            if 'classifier.1' in all_layers:
                relevant.append('classifier.1')
        
        elif 'shufflenet' in arch_type:
            # ShuffleNet: Keep stem, stage blocks, head
            if 'conv1' in all_layers:
                relevant.append('conv1')
            
            for layer in all_layers:
                # Keep stage blocks (e.g., stage2.0, stage3.1)
                if layer.startswith('stage') and layer.count('.') <= 1:
                    relevant.append(layer)
            
            if 'conv5' in all_layers:
                relevant.append('conv5')
        
        elif 'densenet' in arch_type:
            # DenseNet: Prefer block-level outputs + transitions + head
            cifar_block_names = ['conv1', 'dense1', 'trans1', 'dense2', 'trans2', 'dense3', 'bn']
            tv_block_names = [
                'features.conv0', 'features.denseblock1', 'features.transition1',
                'features.denseblock2', 'features.transition2', 'features.denseblock3',
                'features.norm5'
            ]
            
            if any(name in all_layers for name in cifar_block_names):
                for name in cifar_block_names:
                    if name in all_layers:
                        relevant.append(name)
            else:
                for name in tv_block_names:
                    if name in all_layers:
                        relevant.append(name)
        
        else:
            # Unknown architecture - return all layers
            return all_layers
        
        # Remove duplicates while preserving order
        seen = set()
        filtered = []
        for layer in relevant:
            if layer not in seen:
                seen.add(layer)
                filtered.append(layer)
        
        return filtered
    
    def get_conv_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Extract all convolutional layers from a model.
        Works with ResNet, VGG, DenseNet, MobileNet, etc.
        
        For custom CNNs (NarrowCNN, TinyCNN) that have Sequential blocks (conv1, conv2, etc.),
        we extract the blocks themselves rather than individual Conv2d layers inside them.
        """
        layers = {}
        
        # DenseNet CIFAR: use block-level modules
        if (hasattr(model, 'conv1') and hasattr(model, 'dense1') and hasattr(model, 'trans1') and
                hasattr(model, 'dense2') and hasattr(model, 'trans2') and hasattr(model, 'dense3')):
            for name in ('conv1', 'dense1', 'trans1', 'dense2', 'trans2', 'dense3', 'bn'):
                module = getattr(model, name, None)
                if module is not None:
                    layers[name] = module
            return layers
        
        # Torchvision DenseNet: use block-level modules under features
        if (hasattr(model, 'features') and hasattr(model.features, 'denseblock1') and
                hasattr(model.features, 'transition1') and hasattr(model.features, 'denseblock2') and
                hasattr(model.features, 'transition2') and hasattr(model.features, 'denseblock3')):
            for name in ('conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'norm5'):
                module = getattr(model.features, name, None)
                if module is not None:
                    layers[f"features.{name}"] = module
            return layers
        
        # Check if this is a custom CNN with conv1, conv2, etc. Sequential blocks
        has_conv_blocks = (hasattr(model, 'conv1') and isinstance(model.conv1, nn.Sequential) and
                          hasattr(model, 'conv2') and isinstance(model.conv2, nn.Sequential))
        
        if has_conv_blocks:
            # Extract the Sequential blocks directly (conv1, conv2, conv3, conv4)
            for name, module in model.named_children():
                if name.startswith('conv') and isinstance(module, nn.Sequential):
                    # Hook the ENTIRE Sequential block, not just the Conv2d inside
                    # This captures output after Conv2d + BN + ReLU + MaxPool
                    layers[name] = module
        elif (hasattr(model, 'stage2') and hasattr(model, 'stage3') and hasattr(model, 'stage4') and
              isinstance(getattr(model, 'stage2'), nn.Sequential)):
            # ShuffleNet-like models: capture stage blocks and stem/head
            for name in ("conv1", "stage2", "stage3", "stage4", "conv5"):
                module = getattr(model, name, None)
                if isinstance(module, nn.Sequential):
                    layers[name] = module
        elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
            # MobileNet-like models: capture top-level feature blocks (avoid inner convs)
            block_names = {"MobileNetV3Block", "InvertedResidual", "Conv2dNormActivation"}
            is_mobilenet_like = any(m.__class__.__name__ in block_names for m in model.features)
            if is_mobilenet_like:
                for idx, module in enumerate(model.features):
                    if isinstance(module, nn.Conv2d) or module.__class__.__name__ in block_names:
                        layers[f"features.{idx}"] = module
            else:
                # Standard extraction: get all Conv2d layers
                for name, module in model.named_modules():
                    # Only include Conv2d layers
                    if isinstance(module, nn.Conv2d):
                        # Skip if name is empty
                        if name:
                            layers[name] = module
        else:
            # Standard extraction: get all Conv2d layers
            for name, module in model.named_modules():
                # Only include Conv2d layers
                if isinstance(module, nn.Conv2d):
                    # Skip if name is empty
                    if name:
                        layers[name] = module
        
        return layers
    
    def get_linear_layers(self, model: nn.Module, include_intermediate: bool = False) -> Dict[str, nn.Module]:
        """
        Extract linear layers from a model.
        
        Args:
            model: Neural network model
            include_intermediate: If True, include all intermediate layers 
                                (BatchNorm, ReLU, Dropout) in addition to Linear layers.
                                If False, only include Linear layers.
        
        Works with fully-connected networks.
        """
        layers = {}
        
        if include_intermediate:
            # Include ALL layers in the network (Linear, BatchNorm, ReLU, Dropout, etc.)
            # This gives a much denser comparison matrix
            for name, module in model.named_modules():
                if name and '.' in name:  # Skip the top-level 'network' module
                    # Only include layers that produce meaningful representations
                    if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout)):
                        layers[name] = module
        else:
            # Only include Linear layers (default behavior)
            for name, module in model.named_modules():
                # Only include Linear layers
                if isinstance(module, nn.Linear):
                    # Skip if name is empty
                    if name:
                        layers[name] = module
        
        return layers
    
    def get_all_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Extract all Conv2d and Linear layers from a model.
        Useful for mixed model comparisons.
        """
        layers = {}
        
        for name, module in model.named_modules():
            # Include both Conv2d and Linear layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip if name is empty
                if name:
                    layers[name] = module
        
        return layers
    
    def _find_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find a layer by name in the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def get_layer_output_shape(self, model: nn.Module, layer_name: str,
                              input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get output shape of a specific layer."""
        
        model.eval()
        activation = None
        
        # Find and hook the layer
        layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer = module
                break
        
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found")
        
        def hook(module, input, output):
            nonlocal activation
            activation = output
        
        handle = layer.register_forward_hook(hook)
        
        # Forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            _ = model(dummy_input)
        
        handle.remove()
        
        if activation is None:
            raise RuntimeError(f"Failed to capture activation for {layer_name}")
        
        return tuple(activation.shape[1:])  # Return (C, H, W)
    
    def create_and_optimize_stitcher(self, source_model: nn.Module, target_model: nn.Module,
                                    source_layer: str, target_layer: str,
                                    train_loader: DataLoader, input_shape: Tuple[int, ...],
                                    num_epochs: int = 10, lr: float = 1e-3,
                                    max_samples: int = None,
                                    verbose: bool = True,
                                    precomputed_similarity: dict = None) -> Tuple[float, float, float, dict, nn.Module]:
        """
        Create a stitcher and optimize it using cross-entropy loss.
        
        Args:
            source_model: Source model
            target_model: Target model
            source_layer: Name of source layer
            target_layer: Name of target layer
            train_loader: DataLoader for training
            input_shape: Shape of input (C, H, W)
            num_epochs: Number of training epochs
            lr: Learning rate
            max_samples: Maximum number of samples to use (None = use all)
            verbose: Print detailed progress
            precomputed_similarity: Pre-computed similarity metrics to avoid redundant computation
            verbose: Print progress
        
        Returns:
            accuracy_ratio: Stitched accuracy / Target model accuracy
            stitched_accuracy: Accuracy after stitching
            target_accuracy: Original target model accuracy
            similarity_metrics: Dict with CKA, RSA, SVCCA scores
            stitcher: The optimized stitcher module
        """
        
        # Get shapes
        source_shape = self.get_layer_output_shape(source_model, source_layer, input_shape)
        target_shape = self.get_layer_output_shape(target_model, target_layer, input_shape)
        
        # Handle different layer types
        # Conv layers: shape is (C, H, W)
        # Linear layers: shape is (features,)
        
        is_source_conv = len(source_shape) == 3
        is_target_conv = len(target_shape) == 3
        
        if is_source_conv and is_target_conv:
            # Both are convolutional
            source_channels = source_shape[0]
            target_channels = target_shape[0]
            target_spatial = target_shape[1:]
            
            if verbose:
                print(f"  Stitching: {source_layer}[{source_shape}] -> {target_layer}[{target_shape}]")
            
        elif not is_source_conv and not is_target_conv:
            # Both are linear
            source_channels = source_shape[0]
            target_channels = target_shape[0]
            target_spatial = (1, 1)  # Linear layers have no spatial dimensions
            
            if verbose:
                print(f"  Stitching: {source_layer}[{source_channels}] -> {target_layer}[{target_channels}]")
        
        else:
            # Mixed: Conv to Linear or Linear to Conv
            if verbose:
                print(f"  Skipping: incompatible layer types Conv<->Linear {source_shape} -> {target_shape}")
            return 0.0, 0.0, 0.0, {}, None

        # Create appropriate stitcher based on layer type and stitcher_type
        if is_source_conv and is_target_conv:
            # Convolutional layers: choose based on stitcher_type
            if self.stitcher_type == 'affine':
                stitcher = ConvStitcher(source_channels, target_channels, target_spatial).to(self.device)
            elif self.stitcher_type == 'orthogonal':
                stitcher = OrthogonalConvStitcher(source_channels, target_channels, target_spatial).to(self.device)
            elif self.stitcher_type == 'orthogonal_scaled':
                stitcher = OrthogonalScaledConvStitcher(source_channels, target_channels, target_spatial).to(self.device)
            else:
                raise ValueError(f"Unknown stitcher_type: {self.stitcher_type}")
        else:  # Both linear
            # Choose between affine, orthogonal, and orthogonal_scaled stitcher
            if self.stitcher_type == 'affine':
                stitcher = LinearStitcher(source_channels, target_channels).to(self.device)
            elif self.stitcher_type == 'orthogonal':
                stitcher = OrthogonalStitcher(source_channels, target_channels).to(self.device)
            elif self.stitcher_type == 'orthogonal_scaled':
                stitcher = OrthogonalScaledStitcher(source_channels, target_channels).to(self.device)
            else:
                raise ValueError(f"Unknown stitcher_type: {self.stitcher_type}")
        
        # Create stitched network
        stitched_net = StitchedNetwork(
            source_model, target_model, source_layer, target_layer, stitcher
        ).to(self.device)
        
        # Optimization setup
        optimizer = optim.Adam(stitcher.parameters(), lr=lr)
        if self.train_loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = None
        
        # Set models to eval, stitcher to train
        source_model.eval()
        target_model.eval()
        stitcher.train()
        amp_enabled = bool(
            self.use_amp and str(self.device).startswith('cuda') and torch.cuda.is_available()
        )
        if isinstance(stitcher, (InvertibleAffineStitcher, InvertibleAffineConvStitcher)):
            amp_enabled = False
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_entropy_stitched = 0.0
            epoch_entropy_target = 0.0
            num_batches = 0
            samples_processed = 0
            
            # No progress bar
            for inputs, labels in train_loader:
                # Check if we've reached max_samples
                if max_samples is not None and samples_processed >= max_samples:
                    break
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Limit batch size if needed to not exceed max_samples
                if max_samples is not None:
                    remaining = max_samples - samples_processed
                    if remaining < inputs.size(0):
                        inputs = inputs[:remaining]
                        labels = labels[:remaining]
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # Forward pass through stitched network
                    outputs = stitched_net(inputs)
                    
                    # Compute training loss
                    if self.train_loss == 'ce':
                        loss = criterion(outputs, labels)
                    else:
                        with torch.no_grad():
                            target_logits = target_model(inputs)
                        temperature = self.kl_temperature
                        log_p = F.log_softmax(outputs / temperature, dim=1)
                        q = F.softmax(target_logits / temperature, dim=1)
                        kl_loss = F.kl_div(log_p, q, reduction='batchmean') * (temperature ** 2)
                        loss = kl_loss
                        # Entropy ratio logging: H(stitched) / H(target)
                        p = log_p.exp()
                        entropy_stitched = -(p * log_p).sum(dim=1).mean()
                        log_q = F.log_softmax(target_logits / temperature, dim=1)
                        entropy_target = -(q * log_q).sum(dim=1).mean()
                        epoch_kl += kl_loss.item()
                        epoch_entropy_stitched += entropy_stitched.item()
                        epoch_entropy_target += entropy_target.item()
                
                # Add orthogonality regularization for orthogonal-based stitchers
                if isinstance(stitcher, (OrthogonalStitcher, OrthogonalScaledStitcher,
                                        OrthogonalConvStitcher, OrthogonalScaledConvStitcher)):
                    ortho_loss = stitcher.orthogonality_loss()
                    # Weight the orthogonality constraint
                    # Higher weight = stricter orthogonality (0.1 is reasonable)
                    loss = loss + 0.1 * ortho_loss
                
                # Backward pass
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                samples_processed += inputs.size(0)
            
            avg_loss = epoch_loss / num_batches
            avg_kl = epoch_kl / num_batches if self.train_loss == 'kl' else None
            avg_entropy_stitched = epoch_entropy_stitched / num_batches if self.train_loss == 'kl' else None
            avg_entropy_target = epoch_entropy_target / num_batches if self.train_loss == 'kl' else None
            if (self.train_loss == 'kl' and avg_entropy_target and
                    avg_entropy_target > 0.0 and avg_entropy_stitched is not None):
                ent_ratio = 1.0 - (avg_kl / avg_entropy_stitched)
            else:
                ent_ratio = None
            # Store final loss for reporting
            final_loss = avg_loss
            
            if verbose:
                if ent_ratio is not None and self.train_loss == 'kl':
                    print(f"    [Train] Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}, ent_ratio={ent_ratio:.6f}")
                else:
                    print(f"    [Train] Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}")
        
        # Evaluate accuracy after training
        stitcher.eval()
        source_model.eval()
        target_model.eval()
        
        # Decide if we need to collect features (only if computing similarity)
        need_features = precomputed_similarity is None
        
        # Silent evaluation
        
        stitched_correct = 0
        target_correct = 0
        total = 0
        
        # Cross-entropy loss accumulators
        stitched_ce_loss = 0.0
        target_ce_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')  # Sum for averaging later
        kl_sum = 0.0
        entropy_sum = 0.0
        
        # Collect features for similarity metrics (only if needed)
        if need_features:
            source_features_list = []
            target_features_list = []
            stitched_features_list = []
            
            # Hook to capture source features
            source_hook_output = None
            def capture_source_hook(module, input, output):
                nonlocal source_hook_output
                source_hook_output = output.clone()
            
            source_layer_module = self._find_layer(source_model, source_layer)
            source_handle = source_layer_module.register_forward_hook(capture_source_hook)
            
            # Hook to capture target features
            target_hook_output = None
            def capture_target_hook(module, input, output):
                nonlocal target_hook_output
                target_hook_output = output.clone()
            
            target_layer_module = self._find_layer(target_model, target_layer)
            target_handle = target_layer_module.register_forward_hook(capture_target_hook)
        else:
            # No feature collection needed for similarity, but collect stitched features for repr rank
            source_features_list = []
            target_features_list = []
            stitched_features_list = []
            
            # Hook to capture target features for MSE only
            target_hook_output = None
            def capture_target_hook(module, input, output):
                nonlocal target_hook_output
                target_hook_output = output.clone()
            
            target_layer_module = self._find_layer(target_model, target_layer)
            target_handle = target_layer_module.register_forward_hook(capture_target_hook)
            
            mse_sum = 0.0
            var_sum = 0.0
        
        samples_evaluated = 0
        
        with torch.no_grad():
            for inputs, labels in train_loader:
                # Check if we've reached max_samples
                if max_samples is not None and samples_evaluated >= max_samples:
                    break
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Limit batch size if needed to not exceed max_samples
                if max_samples is not None:
                    remaining = max_samples - samples_evaluated
                    if remaining < inputs.size(0):
                        inputs = inputs[:remaining]
                        labels = labels[:remaining]
                
                # Stitched network accuracy and loss
                stitched_outputs = stitched_net(inputs)
                _, stitched_pred = torch.max(stitched_outputs, 1)
                stitched_correct += (stitched_pred == labels).sum().item()
                stitched_ce_loss += criterion(stitched_outputs, labels).item()
                
                samples_evaluated += inputs.size(0)
                
                # Capture features only if needed for similarity computation
                if need_features:
                    # Capture source features
                    source_hook_output = None
                    _ = source_model(inputs)
                    if source_hook_output is not None:
                        source_features_list.append(source_hook_output.cpu())
                        # Apply stitcher to get stitched features
                        stitched_feats = stitcher(source_hook_output)
                        stitched_features_list.append(stitched_feats.cpu())
                    
                    # Target model accuracy, loss, and features
                    target_hook_output = None
                    target_outputs = target_model(inputs)
                    _, target_pred = torch.max(target_outputs, 1)
                    target_correct += (target_pred == labels).sum().item()
                    target_ce_loss += criterion(target_outputs, labels).item()
                    # KL / entropy(stitched) metric (always computed)
                    temperature = self.kl_temperature
                    log_p = F.log_softmax(stitched_outputs / temperature, dim=1)
                    q = F.softmax(target_outputs / temperature, dim=1)
                    kl_batch = F.kl_div(log_p, q, reduction='batchmean') * (temperature ** 2)
                    entropy_batch = -(log_p.exp() * log_p).sum(dim=1).mean()
                    kl_sum += kl_batch.item() * inputs.size(0)
                    entropy_sum += entropy_batch.item() * inputs.size(0)
                    
                    if target_hook_output is not None:
                        target_features_list.append(target_hook_output.cpu())
                else:
                    # Collect stitched features for representation rank only
                    source_act = stitched_net.source_activation
                    if source_act is not None:
                        stitched_feats = stitcher(source_act).detach().cpu()
                        stitched_features_list.append(stitched_feats)
                        source_features_list.append(source_act.detach().cpu())
                    # Compute target accuracy and collect target features for MSE
                    target_hook_output = None
                    target_outputs = target_model(inputs)
                    _, target_pred = torch.max(target_outputs, 1)
                    target_correct += (target_pred == labels).sum().item()
                    target_ce_loss += criterion(target_outputs, labels).item()
                    # KL / entropy(stitched) metric (always computed)
                    temperature = self.kl_temperature
                    log_p = F.log_softmax(stitched_outputs / temperature, dim=1)
                    q = F.softmax(target_outputs / temperature, dim=1)
                    kl_batch = F.kl_div(log_p, q, reduction='batchmean') * (temperature ** 2)
                    entropy_batch = -(log_p.exp() * log_p).sum(dim=1).mean()
                    kl_sum += kl_batch.item() * inputs.size(0)
                    entropy_sum += entropy_batch.item() * inputs.size(0)
                    
                    if target_hook_output is not None and source_act is not None:
                        target_features_list.append(target_hook_output.detach().cpu())
                        target_feats = target_hook_output
                        stitched_feats = stitcher(source_act)
                        mse_sum += torch.sum((target_feats - stitched_feats) ** 2).item()
                        var_sum += torch.sum(target_feats ** 2).item()
                
                total += labels.size(0)
        
        # Remove hooks if they were registered
        if need_features:
            source_handle.remove()
            target_handle.remove()
        else:
            target_handle.remove()
        
        stitched_accuracy = stitched_correct / total
        target_accuracy = target_correct / total
        
        # Average cross-entropy losses
        stitched_ce = stitched_ce_loss / total
        target_ce = target_ce_loss / total
        
        # Compute CE ratio: target_ce / stitched_ce
        # Higher is better: 1.0 means stitched matches target, >1.0 means stitched is better
        if stitched_ce > 1e-6:
            ce_ratio = target_ce / stitched_ce
        else:
            ce_ratio = 1.0 if target_ce < 1e-6 else 0.0
        
        # Compute accuracy ratio (avoid division by zero)
        if target_accuracy > 0:
            accuracy_ratio = stitched_accuracy / target_accuracy
        else:
            accuracy_ratio = 0.0
        
        # Compute similarity metrics (BEFORE stitching only - shared across all stitchers)
        similarity_metrics = {}
        
        # Use precomputed similarity if provided (to avoid redundant computation)
        if precomputed_similarity is not None:
            similarity_metrics = precomputed_similarity.copy()
        else:
            # Compute similarity metrics from scratch
            try:
                if len(source_features_list) > 0 and len(target_features_list) > 0:
                    # Concatenate all batches
                    source_features = torch.cat(source_features_list, dim=0)
                    target_features = torch.cat(target_features_list, dim=0)
                    stitched_features = torch.cat(stitched_features_list, dim=0)

                    # Compute representational similarity (let the metric decide how to handle conv inputs)
                    if verbose or num_epochs > 1:
                        if source_features.ndim == 4:
                            num_features = source_features.shape[1] * source_features.shape[2] * source_features.shape[3]
                        else:
                            num_features = source_features.shape[1]
                        print(f"      Computing similarity metrics ({num_features:,} features)...", end="", flush=True)
                    
                    sim_computer = RepresentationalSimilarity(device=self.device)
                    
                    # Source vs Target (before stitching)
                    metrics_before = sim_computer.compute_all_metrics(
                        source_features, target_features, aggregation=self.similarity_aggregation,
                        max_features=self.max_features_for_similarity
                    )
                    similarity_metrics['cka_before'] = metrics_before['cka']
                    similarity_metrics['rsa_before'] = metrics_before['rsa']
                    similarity_metrics['svcca_before'] = metrics_before['svcca']
                    similarity_metrics['cca_before'] = metrics_before['cca']
                    similarity_metrics['cca_before'] = metrics_before['cca']
                    similarity_metrics['l2_before'] = metrics_before['l2']
                    similarity_metrics['procrustes_before'] = metrics_before['procrustes']
                    similarity_metrics['orthogonal_scaled_before'] = metrics_before['orthogonal_scaled']
                    similarity_metrics['invertible_affine_before'] = metrics_before['invertible_affine']
                
                    # Flatten for reconstruction MSE and rank computation
                    if source_features.ndim == 4:
                        batch_size = source_features.size(0)
                        source_flat = source_features.view(batch_size, -1)
                        target_flat = target_features.view(batch_size, -1)
                        stitched_flat = stitched_features.view(batch_size, -1)
                    else:
                        source_flat = source_features
                        target_flat = target_features
                        stitched_flat = stitched_features
                
                    # Compute normalized MSE between target and stitched features
                    # This measures how well the stitcher transforms source to match target
                    # Formula: ||target - stitched||² / ||target||²
                    mse = torch.mean((target_flat - stitched_flat) ** 2).item()
                    variance = torch.mean(target_flat ** 2).item()
                    if variance > 1e-10:
                        normalized_mse = mse / variance
                    else:
                        normalized_mse = 0.0
                    similarity_metrics['reconstruction_mse'] = normalized_mse
                
                    # Compute numerical rank of representations
                    try:
                        source_rank = self.compute_representation_rank(
                            source_flat, max_features=self.max_features_for_rank
                        )
                        target_rank = self.compute_representation_rank(
                            target_flat, max_features=self.max_features_for_rank
                        )
                        stitched_rank = self.compute_representation_rank(
                            stitched_flat, max_features=self.max_features_for_rank
                        )
                        similarity_metrics['source_repr_rank'] = source_rank
                        similarity_metrics['target_repr_rank'] = target_rank
                        similarity_metrics['stitched_repr_rank'] = stitched_rank
                    except Exception as rank_error:
                        if verbose:
                            print(f"    Warning: Could not compute representation ranks: {rank_error}")
                        similarity_metrics['source_repr_rank'] = 0
                        similarity_metrics['target_repr_rank'] = 0
                        similarity_metrics['stitched_repr_rank'] = 0
                
                    if verbose:
                        print(f"    Similarity: CKA={metrics_before['cka']:.4f}, RSA={metrics_before['rsa']:.4f}, SVCCA={metrics_before['svcca']:.4f}, CCA={metrics_before['cca']:.4f}")
                        print(f"                L2={metrics_before['l2']:.4f}, Procrustes={metrics_before['procrustes']:.4f}")
                        print(f"                Orth+Scale={metrics_before['orthogonal_scaled']:.4f}")
                        print(f"    Reconstruction MSE: {normalized_mse:.4f}")
                    
                    # Print representation ranks if computed successfully
                        src_rank = similarity_metrics.get('source_repr_rank', 0)
                        tgt_rank = similarity_metrics.get('target_repr_rank', 0)
                        stch_rank = similarity_metrics.get('stitched_repr_rank', 0)
                    # Always print, even if 0 (to show they were computed)
                        print(f"    Representation ranks: Source={src_rank}, Target={tgt_rank}, Stitched={stch_rank}")
            except Exception as e:
                # ALWAYS print error (not just when verbose)
                print(f"    ⚠️  Similarity computation failed: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                similarity_metrics = {
                    'cka_before': 0.0, 'rsa_before': 0.0, 'svcca_before': 0.0, 'cca_before': 0.0,
                    'l2_before': 0.0, 'procrustes_before': 0.0,
                    'orthogonal_scaled_before': 0.0,
                    'reconstruction_mse': 0.0,
                    'source_repr_rank': 0, 'target_repr_rank': 0, 'stitched_repr_rank': 0
                }

        # If similarity was skipped, compute stitched representation rank only
        if not need_features and len(stitched_features_list) > 0:
            stitched_features = torch.cat(stitched_features_list, dim=0)
            if stitched_features.ndim == 4:
                batch_size = stitched_features.size(0)
                stitched_features = stitched_features.view(batch_size, -1)
            num_features = stitched_features.size(1)
            stitched_rank = self.compute_representation_rank(
                stitched_features, max_features=self.max_features_for_rank
            )
            similarity_metrics['stitched_repr_rank'] = stitched_rank
            
            if len(source_features_list) > 0 and len(target_features_list) > 0:
                source_features = torch.cat(source_features_list, dim=0)
                target_features = torch.cat(target_features_list, dim=0)
                if source_features.ndim == 4:
                    batch_size = source_features.size(0)
                    source_features = source_features.view(batch_size, -1)
                if target_features.ndim == 4:
                    batch_size = target_features.size(0)
                    target_features = target_features.view(batch_size, -1)
            
            if var_sum > 1e-10:
                similarity_metrics['reconstruction_mse'] = mse_sum / var_sum
            else:
                similarity_metrics['reconstruction_mse'] = 0.0
        
        if verbose:
            print(f"    Stitched Acc: {stitched_accuracy:.4f}, Target Acc: {target_accuracy:.4f}, Ratio: {accuracy_ratio:.4f}")
            print(f"    Stitched CE: {stitched_ce:.4f}, Target CE: {target_ce:.4f}, CE Ratio: {ce_ratio:.4f}")
            
            # Diagnostic: Check if stitcher learned identity for same-size transformations
            if isinstance(stitcher, LinearStitcher) and stitcher.linear.in_features == stitcher.linear.out_features:
                with torch.no_grad():
                    W = stitcher.linear.weight.cpu().numpy()
                    I = np.eye(W.shape[0])
                    identity_dist = np.linalg.norm(W - I, 'fro')
                    print(f"    [Self-stitch diagnostic] Distance from identity: {identity_dist:.4f}")
                    if identity_dist > 1.0:
                        print(f"    ⚠️  WARNING: Stitcher did NOT learn identity!")
            
            # Diagnostic: Show learned scale for OrthogonalScaledStitcher
            if isinstance(stitcher, OrthogonalScaledStitcher):
                scale = stitcher.get_scale()
                print(f"    [Orth+Scale diagnostic] Learned scale: {scale:.4f}")
        
        # Store cross-entropy losses and ratio
        similarity_metrics['stitched_ce'] = stitched_ce
        similarity_metrics['target_ce'] = target_ce
        similarity_metrics['ce_ratio'] = ce_ratio
        similarity_metrics['final_train_loss'] = final_loss
        if entropy_sum > 0.0:
            similarity_metrics['entropy_ratio'] = 1.0 - ((kl_sum / total) / (entropy_sum / total))
        
        # Compute invertibility of the stitcher
        invertibility, numerical_rank, effective_rank, condition_number = self.compute_invertibility(stitcher)
        similarity_metrics['invertibility'] = invertibility
        similarity_metrics['numerical_rank'] = numerical_rank
        similarity_metrics['effective_rank'] = effective_rank
        similarity_metrics['condition_number'] = condition_number
        
        if verbose:
            # Get dimensions for context
            if isinstance(stitcher, LinearStitcher):
                W = stitcher.linear.weight.cpu().numpy()
            elif isinstance(stitcher, (OrthogonalStitcher, OrthogonalScaledStitcher)):
                W = stitcher.weight.cpu().numpy()
            elif isinstance(stitcher, ConvStitcher):
                W = stitcher.conv.weight.cpu().numpy()
                W = W.squeeze()
            elif isinstance(stitcher, (OrthogonalConvStitcher, OrthogonalScaledConvStitcher)):
                W = stitcher.weight.cpu().numpy()
                W = W.squeeze()
            else:
                W = None
            
            if W is not None:
                out_dim, in_dim = W.shape
                print(f"[DEBUG] Stitcher matrix shape: ({out_dim}, {in_dim})")
                print(f"[DEBUG] Max possible rank: {min(out_dim, in_dim)}")
                print(f"    Invertibility: {invertibility:.3f}, Rank: {numerical_rank}/{min(out_dim, in_dim)}, EffRank: {effective_rank:.3f}, Cond: {condition_number:.3g}")
            else:
                print(f"    Invertibility: {invertibility:.3f}, Rank: {numerical_rank}, EffRank: {effective_rank:.3f}, Cond: {condition_number:.3g}")
        
        return accuracy_ratio, stitched_accuracy, target_accuracy, similarity_metrics, stitcher
    
    def compute_representation_rank(self, features: torch.Tensor, max_features: Optional[int] = None) -> int:
        """
        Compute numerical rank of a representation matrix (batch of features).
        
        Args:
            features: Tensor of shape (batch_size, feature_dim) or (batch_size, channels, H, W)
            max_features: If set and feature_dim is larger, use random projection for an approximate rank.
        
        Returns:
            Numerical rank of the feature matrix
        """
        # Flatten spatial dimensions if needed
        if features.ndim == 4:  # (batch, channels, H, W)
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1)  # (B, C*H*W)
        elif features.ndim == 3:  # (batch, channels, spatial)
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1)  # (B, C*spatial)
        
        # Now features should be (batch_size, feature_dim)
        X = features.cpu().numpy()
        
        if max_features is not None and X.shape[1] > max_features:
            # Approximate rank via random projection to reduce feature dimension
            # Use a fixed seed for repeatability across runs
            rng = np.random.default_rng(0)
            proj = rng.standard_normal((X.shape[1], max_features)).astype(X.dtype)
            proj /= np.sqrt(max_features)
            X = X @ proj
        
        try:
            from scipy.linalg import svd
            # Compute SVD of feature matrix
            U, S, Vt = svd(X, full_matrices=False)
            
            # Compute threshold: ε = σ_max × 1e-4
            m, n = X.shape
            eps = S[0] * 1e-4
            rank = np.sum(S > eps)
            
            return int(rank)
        except:
            return 0



    
    def compute_invertibility(self, stitcher) -> float:
        """
        Compute invertibility of the stitcher transformation using condition number.
        For rectangular matrices (dimension mismatch), penalizes information loss.
        
        Args:
            stitcher: Trained stitcher (LinearStitcher, InvertibleAffineStitcher, OrthogonalStitcher,
                     OrthogonalScaledStitcher, ConvStitcher, or InvertibleAffineConvStitcher)
        
        Returns:
            Invertibility score in [0, 1]:
            - 1.0 = Perfectly invertible (well-conditioned, κ=1, square matrix)
            - 0.5 = Moderately invertible (κ=10 or moderate dimension reduction)
            - 0.0 = Singular/poorly invertible (κ→∞ or severe dimension reduction)
            Also returns numerical rank, effective rank, and condition number.
        """
        with torch.no_grad():
            # Extract weight matrix based on stitcher type
            if isinstance(stitcher, LinearStitcher):
                W = stitcher.linear.weight.cpu().numpy()
            elif isinstance(stitcher, InvertibleAffineStitcher):
                W = torch.matrix_exp(stitcher.log_weight).cpu().numpy()
            elif isinstance(stitcher, (OrthogonalStitcher, OrthogonalScaledStitcher)):
                W = stitcher.weight.cpu().numpy()
                # For OrthogonalScaledStitcher, include the scale in the analysis
                if isinstance(stitcher, OrthogonalScaledStitcher):
                    scale = stitcher.get_scale()
                    W = W * scale  # Scale the matrix for invertibility analysis
            elif isinstance(stitcher, ConvStitcher):
                # For ConvStitcher, reshape 1x1 conv to matrix
                W = stitcher.conv.weight.cpu().numpy()  # (out_ch, in_ch, 1, 1)
                W = W.squeeze()  # (out_ch, in_ch)
            elif isinstance(stitcher, InvertibleAffineConvStitcher):
                W = torch.matrix_exp(stitcher.log_weight).cpu().numpy()
            elif isinstance(stitcher, (OrthogonalConvStitcher, OrthogonalScaledConvStitcher)):
                # For orthogonal conv stitchers, extract weight and project to orthogonal
                W_tensor = stitcher.weight.squeeze(-1).squeeze(-1)  # (out_ch, in_ch)
                Q, R = torch.linalg.qr(W_tensor.cpu())
                W = Q.numpy()
                # For scaled version, include the scale
                if isinstance(stitcher, OrthogonalScaledConvStitcher):
                    scale = stitcher.scale.item()
                    W = W * scale
            else:
                return 0.0, 0, 0.0, float('inf')
            
            # Get dimensions
            out_dim, in_dim = W.shape
            
            # Compute SVD
            try:
                from scipy.linalg import svd
                U, S, Vt = svd(W, full_matrices=False)
            except:
                return 0.0, 0, 0.0, float('inf')
            
            # Compute numerical rank (number of singular values above threshold)
            # Threshold: ε = σ_max × 1e-4
            eps = S[0] * 1e-4
            numerical_rank = np.sum(S > eps)
            
            # Compute effective rank (continuous measure of dimensionality)
            # Normalized to [0, 1] where 1 = uses all dimensions equally
            effective_rank = np.sum(S / S.max())
            min_dim = min(in_dim, out_dim)
            normalized_effective_rank = effective_rank / min_dim
            
            # For rectangular matrices, check rank deficiency
            if out_dim != in_dim:
                # Dimension reduction: penalize based on information loss
                min_dim = min(in_dim, out_dim)
                max_dim = max(in_dim, out_dim)
                
                # Rank ratio: what fraction of dimensions are preserved?
                rank_ratio = min_dim / max_dim
                
                # If reducing dimensions significantly, invertibility should be low
                # e.g., 2048 -> 512 gives rank_ratio = 0.25 -> base_invertibility ≈ 0.25
                base_invertibility = rank_ratio
            else:
                # Square matrix: no dimension penalty
                base_invertibility = 1.0
            
            # Compute condition number for the preserved dimensions
            if S[-1] > 1e-10:  # Avoid division by zero
                condition_number = S[0] / S[-1]
            else:
                return 0.0, 0, 0.0, float('inf')  # Singular matrix
            
            # Convert condition number to score
            if condition_number <= 1.0:
                condition_score = 1.0
            else:
                # Use log10 scale for better range distribution
                condition_score = 1.0 / (1.0 + np.log10(condition_number))
            
            # Combine dimension penalty with condition number
            # Both must be good for high invertibility
            invertibility = base_invertibility * condition_score
            
            return float(np.clip(invertibility, 0.0, 1.0)), int(numerical_rank), float(np.clip(normalized_effective_rank, 0.0, 1.0)), float(condition_number)
    
    def collect_layer_features(self, source_model: nn.Module, target_model: nn.Module,
                               source_layer: str, target_layer: str,
                               train_loader: DataLoader, max_samples: int = None,
                               verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect features from source and target layers for similarity computation.
        
        Args:
            source_model: Source model
            target_model: Target model  
            source_layer: Name of source layer
            target_layer: Name of target layer
            train_loader: DataLoader
            max_samples: Maximum samples to collect
            verbose: Print progress
            
        Returns:
            source_features: Tensor of shape (samples, features)
            target_features: Tensor of shape (samples, features)
        """
        source_model.eval()
        target_model.eval()
        
        # Silent feature collection
        
        source_features_list = []
        target_features_list = []
        
        # Hook to capture source features
        source_hook_output = None
        def capture_source_hook(module, input, output):
            nonlocal source_hook_output
            source_hook_output = output.clone()
        
        source_layer_module = self._find_layer(source_model, source_layer)
        source_handle = source_layer_module.register_forward_hook(capture_source_hook)
        
        # Hook to capture target features
        target_hook_output = None
        def capture_target_hook(module, input, output):
            nonlocal target_hook_output
            target_hook_output = output.clone()
        
        target_layer_module = self._find_layer(target_model, target_layer)
        target_handle = target_layer_module.register_forward_hook(capture_target_hook)
        
        samples_collected = 0
        
        with torch.no_grad():
            for inputs, labels in train_loader:
                if max_samples is not None and samples_collected >= max_samples:
                    break
                
                inputs = inputs.to(self.device)
                
                # Limit batch size
                if max_samples is not None:
                    remaining = max_samples - samples_collected
                    if remaining < inputs.size(0):
                        inputs = inputs[:remaining]
                
                # Capture source features
                source_hook_output = None
                _ = source_model(inputs)
                if source_hook_output is not None:
                    source_features_list.append(source_hook_output.cpu())
                
                # Capture target features
                target_hook_output = None
                _ = target_model(inputs)
                if target_hook_output is not None:
                    target_features_list.append(target_hook_output.cpu())
                
                samples_collected += inputs.size(0)
        
        # Remove hooks
        source_handle.remove()
        target_handle.remove()
        
        # Concatenate features
        if len(source_features_list) > 0:
            source_features = torch.cat(source_features_list, dim=0)
            target_features = torch.cat(target_features_list, dim=0)
            
            # Return raw features - let similarity computation handle aggregation
            # Don't flatten here! The aggregation method (gap/flatten/spatial_samples) 
            # will be applied when computing similarity metrics
            return source_features, target_features
        else:
            return None, None
    
    def compute_stitching_matrix(self, source_model: nn.Module, target_model: nn.Module,
                                train_loader: DataLoader, input_shape: Tuple[int, ...],
                                num_epochs: int = 10, layer_filter: str = 'conv',
                                max_samples: int = None,
                                max_samples_similarity: int = None,
                                stitcher_types: list = None,
                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, List[str], List[str]]:
        """
        Compute full stitching matrix between two models.
        Supports computing multiple stitcher types for each layer pair.
        
        Args:
            source_model: Source model
            target_model: Target model
            train_loader: Training data loader
            input_shape: Input shape (C, H, W)
            num_epochs: Number of epochs to optimize each stitcher
            layer_filter: Filter for layer types ('conv', 'linear', 'all')
            max_samples: Maximum number of samples for training/evaluation (None = use all)
            max_samples_similarity: Maximum number of samples for similarity computation (None = use max_samples)
            stitcher_types: List of stitcher types ['affine', 'orthogonal', 'orthogonal_scaled']
                            or None for single type
            verbose: Print progress
        
        Returns:
            If stitcher_types is None or single type:
                accuracy_ratio_matrix, stitched_accuracy_matrix, target_accuracy_matrix,
                similarity_matrices, source_layers, target_layers
            
            If stitcher_types has multiple types:
                results_dict with keys for each stitcher type + 'similarity_matrices', 'source_layers', 'target_layers'
        """
        
        # If max_samples_similarity not specified, use max_samples
        if max_samples_similarity is None:
            max_samples_similarity = max_samples
        
        # Determine stitcher types to compute
        if stitcher_types is None:
            stitcher_types = [self.stitcher_type]
        
        # Get layers based on filter type
        if layer_filter == 'conv':
            source_layers_dict = self.get_conv_layers(source_model)
            target_layers_dict = self.get_conv_layers(target_model)
            layer_type_name = "convolutional"
        elif layer_filter == 'linear':
            source_layers_dict = self.get_linear_layers(source_model)
            target_layers_dict = self.get_linear_layers(target_model)
            layer_type_name = "linear"
        elif layer_filter == 'all':
            source_layers_dict = self.get_all_layers(source_model)
            target_layers_dict = self.get_all_layers(target_model)
            layer_type_name = "conv/linear"
        else:
            raise ValueError(f"Unknown layer_filter: {layer_filter}. Choose from: 'conv', 'linear', 'all'")
        
        # Filter layers - keep all layers
        source_layers = list(source_layers_dict.keys())
        target_layers = list(target_layers_dict.keys())
        
        # Remove empty module names and very short names (usually not real layers)
        source_layers = [name for name in source_layers if name and len(name) > 0]
        target_layers = [name for name in target_layers if name and len(name) > 0]
        
        # Apply block output filtering if enabled
        if self.use_block_outputs:
            total_source_modules = sum(1 for name, _ in source_model.named_modules() if name)
            total_target_modules = sum(1 for name, _ in target_model.named_modules() if name)
            original_source_count = len(source_layers)
            original_target_count = len(target_layers)
            
            source_layers = self.filter_block_outputs(source_layers)
            target_layers = self.filter_block_outputs(target_layers)
            
            if verbose:
                print(f"\n[Block Output Filtering]")
                print(f"  Source: {total_source_modules} total → {original_source_count} {layer_type_name} → {len(source_layers)} block outputs")
                print(f"  Target: {total_target_modules} total → {original_target_count} {layer_type_name} → {len(target_layers)} block outputs")
                print(f"  Comparisons: {original_source_count * original_target_count} → {len(source_layers) * len(target_layers)}")
        
        if len(source_layers) == 0:
            raise ValueError(f"No {layer_type_name} layers found in source model")
        if len(target_layers) == 0:
            raise ValueError(f"No {layer_type_name} layers found in target model")
        
        if verbose:
            print(f"\nSource layers ({len(source_layers)}): {source_layers}")
            print(f"Target layers ({len(target_layers)}): {target_layers}")
            if stitcher_types and len(stitcher_types) > 1:
                print(f"Stitcher types: {', '.join(stitcher_types)}\n")
            else:
                print()
        
        # Initialize matrices for each stitcher type
        if stitcher_types and len(stitcher_types) > 1:
            results = {}
            for st_type in stitcher_types:
                results[st_type] = {
                    'accuracy_ratio': np.zeros((len(source_layers), len(target_layers))),
                    'stitched_accuracy': np.zeros((len(source_layers), len(target_layers))),
                    'target_accuracy': np.zeros((len(source_layers), len(target_layers))),
                    'stitched_ce': np.zeros((len(source_layers), len(target_layers))),
                    'target_ce': np.zeros((len(source_layers), len(target_layers))),
                    'ce_ratio': np.zeros((len(source_layers), len(target_layers))),  # NEW
                    'entropy_ratio': np.zeros((len(source_layers), len(target_layers))),
                    'invertibility': np.zeros((len(source_layers), len(target_layers))),
                    'condition_number': np.zeros((len(source_layers), len(target_layers))),
                    'numerical_rank': np.zeros((len(source_layers), len(target_layers)), dtype=int),
                    'effective_rank': np.zeros((len(source_layers), len(target_layers))),
                    'reconstruction_mse': np.zeros((len(source_layers), len(target_layers))),
                    'stitched_repr_rank': np.zeros((len(source_layers), len(target_layers)), dtype=int),
                }
        else:
            # Single stitcher type - use original format
            accuracy_ratio_matrix = np.zeros((len(source_layers), len(target_layers)))
            stitched_accuracy_matrix = np.zeros((len(source_layers), len(target_layers)))
            target_accuracy_matrix = np.zeros((len(source_layers), len(target_layers)))
            stitched_ce_matrix = np.zeros((len(source_layers), len(target_layers)))
            target_ce_matrix = np.zeros((len(source_layers), len(target_layers)))
            ce_ratio_matrix = np.zeros((len(source_layers), len(target_layers)))  # NEW
            entropy_ratio_matrix = np.zeros((len(source_layers), len(target_layers)))
            invertibility_matrix = np.zeros((len(source_layers), len(target_layers)))
            condition_number_matrix = np.zeros((len(source_layers), len(target_layers)))
            numerical_rank_matrix = np.zeros((len(source_layers), len(target_layers)), dtype=int)
            effective_rank_matrix = np.zeros((len(source_layers), len(target_layers)))
            reconstruction_mse_matrix = np.zeros((len(source_layers), len(target_layers)))
            stitched_repr_rank_matrix = np.zeros((len(source_layers), len(target_layers)), dtype=int)
        
        # Initialize similarity metric matrices (BEFORE stitching - shared across all stitchers)
        cka_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        rsa_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        svcca_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        cca_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        l2_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        procrustes_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        orthogonal_scaled_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        invertible_affine_before_matrix = np.zeros((len(source_layers), len(target_layers)))
        
        # Representation rank matrices (shared - only source & target, stitched is per-stitcher)
        source_repr_rank_matrix = np.zeros((len(source_layers), len(target_layers)), dtype=int)
        target_repr_rank_matrix = np.zeros((len(source_layers), len(target_layers)), dtype=int)
        
        # Timing accumulators
        stitcher_time = 0.0
        similarity_time = 0.0
        
        # Compute stitching for each pair
        for i, source_layer in enumerate(source_layers):
            for j, target_layer in enumerate(target_layers):
                # Get shapes for display
                source_shape = self.get_layer_output_shape(source_model, source_layer, input_shape)
                target_shape = self.get_layer_output_shape(target_model, target_layer, input_shape)
                
                if verbose:
                    print(f"\n[{i+1}/{len(source_layers)}][{j+1}/{len(target_layers)}] {source_layer}{list(source_shape)} -> {target_layer}{list(target_shape)}")
                pair_stitcher_time = 0.0
                pair_similarity_time = 0.0
                
                # Determine which stitcher types to run for this pair
                if stitcher_types and len(stitcher_types) > 1:
                    types_to_run = stitcher_types
                else:
                    types_to_run = [self.stitcher_type]
                
                # STEP 1: Collect features ONCE before training any stitchers
                # These features are used for similarity computation later
                # Use max_samples_similarity (can be different from training samples)
                source_features, target_features = self.collect_layer_features(
                    source_model, target_model, source_layer, target_layer,
                    train_loader, max_samples=max_samples_similarity,
                    verbose=verbose or len(types_to_run) > 1
                )
                
                # STEP 2: Train each stitcher type (without collecting features again)
                for st_idx, st_type in enumerate(types_to_run):
                    try:
                        # Temporarily set stitcher type
                        original_stitcher_type = self.stitcher_type
                        self.stitcher_type = st_type
                        
                        # Train stitcher without computing similarity (pass empty dict to skip)
                        start = time.perf_counter()
                        accuracy_ratio, stitched_acc, target_acc, sim_metrics, _ = self.create_and_optimize_stitcher(
                            source_model, target_model,
                            source_layer, target_layer,
                            train_loader, input_shape,
                            num_epochs=num_epochs,
                            max_samples=max_samples,
                            verbose=verbose and len(types_to_run) == 1,
                            precomputed_similarity={}  # Empty dict = skip similarity computation
                        )
                        elapsed = time.perf_counter() - start
                        stitcher_time += elapsed
                        pair_stitcher_time += elapsed
                        
                        # Restore stitcher type
                        self.stitcher_type = original_stitcher_type
                        
                        # Store results
                        if len(types_to_run) > 1:
                            results[st_type]['accuracy_ratio'][i, j] = accuracy_ratio
                            results[st_type]['stitched_accuracy'][i, j] = stitched_acc
                            results[st_type]['target_accuracy'][i, j] = target_acc
                            results[st_type]['stitched_ce'][i, j] = sim_metrics.get('stitched_ce', 0.0)
                            results[st_type]['target_ce'][i, j] = sim_metrics.get('target_ce', 0.0)
                            results[st_type]['ce_ratio'][i, j] = sim_metrics.get('ce_ratio', 0.0)
                            results[st_type]['entropy_ratio'][i, j] = sim_metrics.get('entropy_ratio', 0.0)
                            results[st_type]['invertibility'][i, j] = sim_metrics.get('invertibility', 0.0)
                            results[st_type]['condition_number'][i, j] = sim_metrics.get('condition_number', 0.0)
                            results[st_type]['numerical_rank'][i, j] = sim_metrics.get('numerical_rank', 0)
                            results[st_type]['effective_rank'][i, j] = sim_metrics.get('effective_rank', 0.0)
                            results[st_type]['reconstruction_mse'][i, j] = sim_metrics.get('reconstruction_mse', 0.0)
                            results[st_type]['stitched_repr_rank'][i, j] = sim_metrics.get('stitched_repr_rank', 0)
                        else:
                            accuracy_ratio_matrix[i, j] = accuracy_ratio
                            stitched_accuracy_matrix[i, j] = stitched_acc
                            target_accuracy_matrix[i, j] = target_acc
                            stitched_ce_matrix[i, j] = sim_metrics.get('stitched_ce', 0.0)
                            target_ce_matrix[i, j] = sim_metrics.get('target_ce', 0.0)
                            ce_ratio_matrix[i, j] = sim_metrics.get('ce_ratio', 0.0)
                            entropy_ratio_matrix[i, j] = sim_metrics.get('entropy_ratio', 0.0)
                            invertibility_matrix[i, j] = sim_metrics.get('invertibility', 0.0)
                            condition_number_matrix[i, j] = sim_metrics.get('condition_number', 0.0)
                            numerical_rank_matrix[i, j] = sim_metrics.get('numerical_rank', 0)
                            effective_rank_matrix[i, j] = sim_metrics.get('effective_rank', 0.0)
                            reconstruction_mse_matrix[i, j] = sim_metrics.get('reconstruction_mse', 0.0)
                            stitched_repr_rank_matrix[i, j] = sim_metrics.get('stitched_repr_rank', 0)
                        
                        if verbose and len(types_to_run) > 1:
                            rank_str = sim_metrics.get('stitched_repr_rank', 0)
                            ce_r = sim_metrics.get('ce_ratio', 0.0)
                            inv = sim_metrics.get('invertibility', 0.0)
                            cond = sim_metrics.get('condition_number', 0.0)
                            eff_rank = sim_metrics.get('effective_rank', 0.0)
                            mse = sim_metrics.get('reconstruction_mse', 0.0)
                            train_loss = sim_metrics.get('final_train_loss', 0.0)
                            ent_ratio = sim_metrics.get('entropy_ratio', None)
                            prefix = f"[{st_type.upper()}]"
                            if ent_ratio is not None:
                                print(f"{prefix} Ratio={accuracy_ratio:.3f} CE_R={ce_r:.3f} Inv={inv:.3f} Cond={cond:.3g} Rank={rank_str} EffRank={eff_rank:.3f} MSE={mse:.3f} Ent_R={ent_ratio:.3f} TrainLoss={train_loss:.4f}")
                            else:
                                print(f"{prefix} Ratio={accuracy_ratio:.3f} CE_R={ce_r:.3f} Inv={inv:.3f} Cond={cond:.3g} Rank={rank_str} EffRank={eff_rank:.3f} MSE={mse:.3f} TrainLoss={train_loss:.4f}")
                    
                    except Exception as e:
                        print(f"\n  ⚠️  Exception in stitcher '{st_type}': {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        # Store zero results on error
                        if len(types_to_run) > 1:
                            results[st_type]['accuracy_ratio'][i, j] = 0.0
                        else:
                            accuracy_ratio_matrix[i, j] = 0.0
                
                # STEP 3: Compute similarity metrics ONCE after all stitchers are trained
                # These are independent of stitcher type (compare source vs target only)
                if source_features is not None and target_features is not None:
                    try:
                        # Always compute similarity metrics (fast metrics always, slow metrics conditionally)
                        start = time.perf_counter()
                        sim_computer = RepresentationalSimilarity(device=self.device)
                        metrics = sim_computer.compute_all_metrics(
                            source_features, target_features, aggregation=self.similarity_aggregation,
                            max_features=self.max_features_for_similarity
                        )
                        elapsed = time.perf_counter() - start
                        similarity_time += elapsed
                        pair_similarity_time += elapsed
                        
                        cka_before_matrix[i, j] = metrics['cka']
                        rsa_before_matrix[i, j] = metrics['rsa']
                        svcca_before_matrix[i, j] = metrics['svcca']
                        cca_before_matrix[i, j] = metrics['cca']
                        l2_before_matrix[i, j] = metrics['l2']
                        procrustes_before_matrix[i, j] = metrics['procrustes']
                        orthogonal_scaled_before_matrix[i, j] = metrics['orthogonal_scaled']
                        invertible_affine_before_matrix[i, j] = metrics['invertible_affine']
                        
                        # Compute representation ranks (effective dimensionality)
                        # Need to aggregate and convert to numpy first
                        if source_features.ndim == 4:
                            sim_temp = RepresentationalSimilarity(device=self.device)
                            source_agg = sim_temp._aggregate_spatial(source_features, self.similarity_aggregation)
                            target_agg = sim_temp._aggregate_spatial(target_features, self.similarity_aggregation)
                        else:
                            source_agg = source_features
                            target_agg = target_features
                        
                        X_np = source_agg.cpu().numpy()
                        Y_np = target_agg.cpu().numpy()
                        X_centered = X_np - X_np.mean(axis=0, keepdims=True)
                        Y_centered = Y_np - Y_np.mean(axis=0, keepdims=True)
                        
                        source_repr_rank_matrix[i, j] = self.compute_representation_rank(
                            source_features, max_features=self.max_features_for_rank
                        )
                        target_repr_rank_matrix[i, j] = self.compute_representation_rank(
                            target_features, max_features=self.max_features_for_rank
                        )
                        
                    
                    except Exception as e:
                        # Silent failure - just set defaults
                        cka_before_matrix[i, j] = 0.0
                        rsa_before_matrix[i, j] = 0.0
                        svcca_before_matrix[i, j] = 0.0
                        cca_before_matrix[i, j] = 0.0
                        l2_before_matrix[i, j] = 0.0
                        procrustes_before_matrix[i, j] = 0.0
                        orthogonal_scaled_before_matrix[i, j] = 0.0
                        invertible_affine_before_matrix[i, j] = 0.0
                
                # Print similarity results (once) with repr ranks
                if verbose:
                    print(f"  Similarity: CKA={cka_before_matrix[i, j]:.4f}, RSA={rsa_before_matrix[i, j]:.4f}, SVCCA={svcca_before_matrix[i, j]:.4f}, CCA={cca_before_matrix[i, j]:.4f}, L2={l2_before_matrix[i, j]:.4f}, Proc={procrustes_before_matrix[i, j]:.4f}, Orth+Sc={orthogonal_scaled_before_matrix[i, j]:.4f}, InvAff={invertible_affine_before_matrix[i, j]:.4f}")
                    
                    # Print representation ranks
                    if len(types_to_run) > 1:
                        print(f"  Repr ranks: Source={source_repr_rank_matrix[i,j]}, Target={target_repr_rank_matrix[i,j]}")
                    print()  # Blank line
        
        similarity_matrices = {
            'cka_before': cka_before_matrix,
            'rsa_before': rsa_before_matrix,
            'svcca_before': svcca_before_matrix,
            'cca_before': cca_before_matrix,
            'l2_before': l2_before_matrix,
            'procrustes_before': procrustes_before_matrix,
            'orthogonal_scaled_before': orthogonal_scaled_before_matrix,
            'invertible_affine_before': invertible_affine_before_matrix,
            'source_repr_rank': source_repr_rank_matrix,
            'target_repr_rank': target_repr_rank_matrix,
        }
        
        # Return format depends on number of stitcher types
        if verbose:
            total_time = stitcher_time + similarity_time
            print(f"\nTiming totals: stitchers={stitcher_time:.1f}s, similarity={similarity_time:.1f}s, total={total_time:.1f}s")
        
        if stitcher_types and len(stitcher_types) > 1:
            results['similarity_matrices'] = similarity_matrices
            results['source_layers'] = source_layers
            results['target_layers'] = target_layers
            return results
        else:
            return accuracy_ratio_matrix, stitched_accuracy_matrix, target_accuracy_matrix, entropy_ratio_matrix, similarity_matrices, source_layers, target_layers


def example_usage():
    """Example of how to use the improved stitcher."""
    from torchvision import models, datasets, transforms
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stitcher = ImprovedModelStitcher(device=device)
    
    # Load models
    resnet = models.resnet18(pretrained=True).to(device)
    vgg = models.vgg11(pretrained=True).to(device)
    
    # Adapt for CIFAR-10
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    vgg.classifier[6] = nn.Linear(4096, 10)
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Use small subset for testing
    subset_indices = np.random.choice(len(train_dataset), 500, replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    
    # Compute stitching matrix (ResNet to VGG)
    print("Computing ResNet -> VGG stitching matrix...")
    acc_ratio_matrix, stitched_acc, target_acc, entropy_ratio, similarity_matrices, src_layers, tgt_layers = stitcher.compute_stitching_matrix(
        resnet, vgg, train_loader, input_shape=(3, 224, 224),
        num_epochs=3, verbose=True
    )
    
    # Save results
    np.save('resnet_vgg_accuracy_ratio.npy', acc_ratio_matrix)
    np.save('resnet_vgg_stitched_acc.npy', stitched_acc)
    np.save('resnet_vgg_target_acc.npy', target_acc)
    
    print(f"\nAccuracy Ratio Matrix shape: {acc_ratio_matrix.shape}")
    print(f"Mean accuracy ratio: {np.mean(acc_ratio_matrix[acc_ratio_matrix > 0]):.4f}")
    print(f"Best accuracy ratio: {np.max(acc_ratio_matrix):.4f}")
    print(f"Stitches with ratio >= 0.9: {np.sum(acc_ratio_matrix >= 0.9)}/{acc_ratio_matrix.size}")


if __name__ == '__main__':
    example_usage()
