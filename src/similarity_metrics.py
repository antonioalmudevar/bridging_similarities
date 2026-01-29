"""
Representational Similarity Metrics for Neural Networks

Implements:
- CKA (Centered Kernel Alignment)
- RSA (Representational Similarity Analysis)
- SVCCA (Singular Vector Canonical Correlation Analysis)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import svd, qr
from typing import Tuple


class RepresentationalSimilarity:
    """Compute representational similarity metrics between layer activations."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def compute_all_metrics(self, features1: torch.Tensor, features2: torch.Tensor,
                          aggregation: str = 'flatten', max_features: int = 2048) -> dict:
        """
        Compute all similarity metrics between two feature representations.
        
        Args:
            features1: (batch, channels, height, width) for conv layers
                      OR (batch, features) for linear layers
            features2: (batch, channels, height, width) for conv layers
                      OR (batch, features) for linear layers
            aggregation: How to handle spatial dimensions (only for conv layers):
                - 'gap': Global Average Pooling
                - 'flatten': Flatten all spatial locations (default)
                - 'spatial_samples': Treat spatial locations as samples
                Note: For linear layers (2D tensors), aggregation is ignored
            max_features: Maximum number of features for slow metrics.
                         Fast metrics (CKA, SVCCA): always computed
                         Slow metrics (RSA, L2, Procrustes, Orth+Scale): only if features <= max_features
        
        Returns:
            Dictionary with CKA, SVCCA, CCA, RSA, L2, Procrustes, Orthogonal+Scale,
            and Invertible Affine scores
        """
        
        # Handle spatial dimensions (only if 4D - convolutional)
        is_conv = len(features1.shape) == 4 and len(features2.shape) == 4
        if len(features1.shape) == 4:
            features1_agg = self._aggregate_spatial(features1, aggregation)
        else:
            features1_agg = features1
        if len(features2.shape) == 4:
            features2_agg = self._aggregate_spatial(features2, aggregation)
        else:
            features2_agg = features2
        
        # Convert to numpy for easier computation
        X = features1_agg.detach().cpu().numpy()  # (n_samples, n_features1)
        Y = features2_agg.detach().cpu().numpy()  # (n_samples, n_features2)
        
        # Determine if we should skip slow metrics based on feature count
        num_features = max(X.shape[1], Y.shape[1])
        skip_slow = num_features > max_features
        conv_channels = max(features1.shape[1], features2.shape[1]) if is_conv else None
        skip_conv_align = conv_channels is not None and conv_channels > max_features
        
        results = {}
        
        # Fast metrics (always computed)
        try:
            results['cka'] = self.linear_cka(X, Y)
        except Exception as e:
            print(f"    CKA computation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results['cka'] = 0.0
        
        try:
            results['svcca'] = self.svcca(X, Y)
        except Exception as e:
            print(f"    SVCCA computation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results['svcca'] = 0.0
        
        try:
            results['cca'] = self.cca(X, Y)
        except Exception as e:
            print(f"    CCA computation failed: {type(e).__name__}: {e}")
            results['cca'] = 0.0
        
        # Slow metrics (RSA only)
        if not skip_slow:
            try:
                results['rsa'] = self.rsa(X, Y)
            except Exception as e:
                print(f"    RSA computation failed: {type(e).__name__}: {e}")
                results['rsa'] = 0.0
        else:
            results['rsa'] = 0.0
        
        # Always compute L2/Procrustes/Orth+Scale/Invertible Affine (independent of max_features)
        try:
            if is_conv:
                results['l2'] = self._conv1x1_similarity(features1, features2, mode='affine')
            else:
                results['l2'] = self.l2_regression_similarity(X, Y)
        except Exception as e:
            print(f"    L2 computation failed: {type(e).__name__}: {e}")
            results['l2'] = 0.0
        
        try:
            if is_conv:
                results['procrustes'] = self._conv1x1_similarity(features1, features2, mode='orthogonal')
            else:
                results['procrustes'] = self.procrustes_similarity(X, Y)
        except Exception as e:
            print(f"    Procrustes computation failed: {type(e).__name__}: {e}")
            results['procrustes'] = 0.0
        
        try:
            if is_conv:
                results['orthogonal_scaled'] = self._conv1x1_similarity(features1, features2, mode='orthogonal_scaled')
            else:
                results['orthogonal_scaled'] = self.orthogonal_scaled_similarity(X, Y)
        except Exception as e:
            print(f"    Orth+Scale computation failed: {type(e).__name__}: {e}")
            results['orthogonal_scaled'] = 0.0

        try:
            if is_conv:
                results['invertible_affine'] = self._conv1x1_similarity(
                    features1, features2, mode='invertible_affine'
                )
            else:
                results['invertible_affine'] = self.invertible_affine_similarity(X, Y)
        except Exception as e:
            print(f"    Invertible Affine computation failed: {type(e).__name__}: {e}")
            results['invertible_affine'] = 0.0
        
        return results
    
    def _aggregate_spatial(self, features: torch.Tensor, method: str) -> torch.Tensor:
        """
        Aggregate spatial dimensions of 4D feature tensor.
        
        Args:
            features: (batch, channels, height, width) or (batch, features) for linear layers
            method: 'gap', 'flatten', or 'spatial_samples'
        
        Returns:
            2D tensor (n_samples, n_features)
        """
        # If already 2D (linear layer output), return as-is
        if len(features.shape) == 2:
            return features
        
        # Otherwise, handle 4D convolutional features
        if len(features.shape) != 4:
            raise ValueError(f"Expected 2D or 4D features, got shape {features.shape}")
        
        B, C, H, W = features.shape
        
        if method == 'gap':
            # Global Average Pooling: (B, C, H, W) → (B, C)
            # Average over spatial dimensions
            return features.mean(dim=[2, 3])
        
        elif method == 'flatten':
            # Flatten: (B, C, H, W) → (B, C*H*W)
            # Treat each spatial location as a separate feature
            return features.reshape(B, -1)
        
        elif method == 'spatial_samples':
            # Spatial as samples: (B, C, H, W) → (B*H*W, C)
            # Treat each spatial location as a sample
            return features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}. Use 'gap', 'flatten', or 'spatial_samples'")

    def _invertibility_regularizer(self, W: torch.Tensor, singular_floor: float) -> torch.Tensor:
        """Penalize small singular values to encourage full-rank behavior."""
        s = torch.linalg.svdvals(W)
        penalty = F.relu(singular_floor - s)
        return torch.mean(penalty * penalty)

    def _conv1x1_similarity(self, features1: torch.Tensor, features2: torch.Tensor,
                            mode: str, epochs: int = 5, lr: float = 1e-2,
                            batch_size: int = 128, reg_weight: float = 0.01,
                            singular_floor: float = 1e-3) -> float:
        """
        Fit a 1x1 convolution to align 4D features and return 1 - normalized MSE.
        
        mode:
            - 'affine' (unconstrained 1x1 conv with bias)
            - 'orthogonal' (orthogonal 1x1 conv, no bias)
            - 'orthogonal_scaled' (orthogonal 1x1 conv with scalar scale, no bias)
            - 'invertible_affine' (affine 1x1 conv with invertibility regularizer)
        """
        if features1.ndim != 4 or features2.ndim != 4:
            raise ValueError("1x1 conv alignment requires 4D features")
        
        device = self.device
        X = features1.detach().to(device).float()
        Y = features2.detach().to(device).float()
        
        # Match spatial size by pooling source to target size if needed
        if X.shape[2:] != Y.shape[2:]:
            X = F.adaptive_avg_pool2d(X, Y.shape[2:])
        
        in_channels = X.shape[1]
        out_channels = Y.shape[1]
        
        if mode == 'affine':
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).to(device)
            if in_channels == out_channels:
                with torch.no_grad():
                    conv.weight.copy_(torch.eye(in_channels, out_channels, device=device).unsqueeze(-1).unsqueeze(-1))
                    conv.bias.zero_()
            else:
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(conv.bias)
            params = conv.parameters()
            
            def forward(x):
                return conv(x)
            
            def ortho_loss():
                return 0.0
            
            def inv_loss():
                return 0.0
        elif mode == 'invertible_affine':
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).to(device)
            if in_channels == out_channels:
                with torch.no_grad():
                    conv.weight.copy_(torch.eye(in_channels, out_channels, device=device).unsqueeze(-1).unsqueeze(-1))
                    conv.bias.zero_()
            else:
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(conv.bias)
            params = conv.parameters()
            
            def forward(x):
                return conv(x)
            
            def ortho_loss():
                return 0.0
            
            def inv_loss():
                w = conv.weight.squeeze(-1).squeeze(-1)
                return self._invertibility_regularizer(w, singular_floor)
        else:
            weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, device=device))
            if in_channels == out_channels:
                with torch.no_grad():
                    weight.copy_(torch.eye(in_channels, out_channels, device=device).unsqueeze(-1).unsqueeze(-1))
            else:
                with torch.no_grad():
                    w_init = torch.randn(out_channels, in_channels, device=device)
                    if out_channels <= in_channels:
                        q, _ = torch.linalg.qr(w_init.T)
                        w_init = q.T
                    else:
                        q, _ = torch.linalg.qr(w_init)
                        w_init = q
                    weight.copy_(w_init.unsqueeze(-1).unsqueeze(-1))
            
            if mode == 'orthogonal_scaled':
                log_scale = nn.Parameter(torch.zeros(1, device=device))
                params = [weight, log_scale]
                
                def forward(x):
                    return F.conv2d(x, torch.exp(log_scale) * weight, bias=None)
            elif mode == 'orthogonal':
                log_scale = None
                params = [weight]
                
                def forward(x):
                    return F.conv2d(x, weight, bias=None)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            def ortho_loss():
                w = weight.squeeze(-1).squeeze(-1)
                wtw = torch.matmul(w.t(), w)
                n = min(out_channels, in_channels)
                I = torch.eye(n, device=device, dtype=w.dtype)
                if out_channels == in_channels:
                    return torch.norm(wtw - I, p='fro') ** 2
                return torch.norm(wtw[:n, :n] - I, p='fro') ** 2
            
            def inv_loss():
                return 0.0
        
        optimizer = optim.Adam(params, lr=lr)
        num_samples = X.size(0)
        
        for _ in range(epochs):
            perm = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                idx = perm[start:start + batch_size]
                x_batch = X[idx]
                y_batch = Y[idx]
                
                optimizer.zero_grad()
                preds = forward(x_batch)
                loss = F.mse_loss(preds, y_batch)
                if mode in ('orthogonal', 'orthogonal_scaled'):
                    loss = loss + 0.1 * ortho_loss()
                if mode == 'invertible_affine' and reg_weight > 0:
                    loss = loss + reg_weight * inv_loss()
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            preds = forward(X)
            mse = torch.mean((Y - preds) ** 2).item()
            var = torch.mean(Y ** 2).item()
        
        if var > 1e-10:
            similarity = 1.0 - (mse / var)
        else:
            similarity = 1.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Linear CKA (Centered Kernel Alignment).
        
        CKA measures similarity between representations using HSIC.
        Range: [0, 1], where 1 = identical representations
        
        Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations 
                   Revisited"
        
        Args:
            X: (n_samples, n_features1)
            Y: (n_samples, n_features2)
        
        Returns:
            CKA similarity score
        """
        
        # Center the matrices
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # Compute gram matrices
        K = X @ X.T  # (n, n)
        L = Y @ Y.T  # (n, n)
        
        # Center the gram matrices
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H
        L = H @ L @ H
        
        # Compute HSIC
        hsic = np.sum(K * L)
        
        # Normalize
        norm_K = np.sqrt(np.sum(K * K))
        norm_L = np.sqrt(np.sum(L * L))
        
        if norm_K * norm_L == 0:
            return 0.0
        
        cka = hsic / (norm_K * norm_L)
        
        return float(np.clip(cka, 0, 1))
    
    def rsa(self, X: np.ndarray, Y: np.ndarray, method='spearman') -> float:
        """
        Compute RSA (Representational Similarity Analysis).
        
        RSA compares the similarity structure (RDM - representational dissimilarity matrix)
        between two representations.
        
        Range: [-1, 1] for Spearman correlation, where 1 = identical structure
        
        Reference: Kriegeskorte et al. (2008) "Representational similarity analysis"
        
        Args:
            X: (n_samples, n_features1)
            Y: (n_samples, n_features2)
            method: 'spearman' or 'pearson'
        
        Returns:
            RSA similarity score
        """
        
        # Need at least 2 samples for pairwise distances
        if X.shape[0] < 2:
            return 0.0
        
        # Compute pairwise distances (RDMs)
        from scipy.spatial.distance import pdist, squareform
        
        try:
            rdm_X = pdist(X, metric='correlation')  # 1 - correlation
            rdm_Y = pdist(Y, metric='correlation')
        except Exception as e:
            # If pdist fails (e.g., constant features), return 0
            return 0.0
        
        # Compute correlation between RDMs
        if method == 'spearman':
            correlation, _ = spearmanr(rdm_X, rdm_Y)
        else:  # pearson
            correlation = np.corrcoef(rdm_X, rdm_Y)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    def svcca(self, X: np.ndarray, Y: np.ndarray, 
              threshold: float = 0.99) -> float:
        """
        Compute SVCCA (Singular Vector Canonical Correlation Analysis).
        
        SVCCA first applies SVD to reduce dimensionality, then computes CCA.
        
        Range: [0, 1], where 1 = perfectly aligned subspaces
        
        Reference: Raghu et al. (2017) "SVCCA: Singular Vector Canonical 
                   Correlation Analysis for Deep Learning Dynamics and Interpretability"
        
        Args:
            X: (n_samples, n_features1)
            Y: (n_samples, n_features2)
            threshold: Variance threshold for SVD (default: 0.99)
        
        Returns:
            Mean canonical correlation
        """
        
        # Step 1: SVD to reduce dimensionality
        X_reduced = self._svd_reduction(X, threshold)
        Y_reduced = self._svd_reduction(Y, threshold)
        
        # Step 2: Compute CCA
        cca_similarity = self._cca(X_reduced, Y_reduced)
        
        return float(cca_similarity)
    
    def _svd_reduction(self, X: np.ndarray, threshold: float = 0.99) -> np.ndarray:
        """
        Reduce dimensionality using SVD, keeping components that explain 
        `threshold` of variance.
        
        Args:
            X: (n_samples, n_features)
            threshold: Variance explained threshold
        
        Returns:
            Reduced representation
        """
        
        # Center
        X = X - X.mean(axis=0, keepdims=True)
        
        # SVD
        U, S, Vt = svd(X, full_matrices=False)
        
        # Determine number of components
        variance_explained = np.cumsum(S**2) / np.sum(S**2)
        n_components = np.searchsorted(variance_explained, threshold) + 1
        n_components = min(n_components, len(S))
        
        # Project onto top components
        X_reduced = U[:, :n_components] * S[:n_components]
        
        return X_reduced
    
    def _cca(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Canonical Correlation Analysis.
        
        Args:
            X: (n_samples, n_features1)
            Y: (n_samples, n_features2)
        
        Returns:
            Mean canonical correlation
        """
        
        n = X.shape[0]
        
        # Center
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # QR decomposition for numerical stability
        Qx, Rx = qr(X, mode='economic')
        Qy, Ry = qr(Y, mode='economic')
        
        # SVD of Qx.T @ Qy
        U, S, Vt = svd(Qx.T @ Qy, full_matrices=False)
        
        # Canonical correlations are the singular values
        # Clip to [0, 1] range (can be slightly > 1 due to numerical errors)
        canonical_correlations = np.clip(S, 0, 1)
        
        # Return mean correlation
        return np.mean(canonical_correlations)

    def cca(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute mean canonical correlation without SVD truncation.
        """
        return float(self._cca(X, Y))

    def l2_regression_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute L² similarity: 1 - MSE between Y and linear regression prediction Ŷ = XW.
        
        This measures how well Y can be predicted from X using a linear regressor
        trained to minimize MSE. Higher similarity means representations are more linearly related.
        
        The optimal linear transformation is: W = (X^T X)^{-1} X^T Y (least squares)
        
        Args:
            X: Source features (n_samples, n_features1)
            Y: Target features (n_samples, n_features2)
        
        Returns:
            L² similarity: 1 - (||Y - XW||² / ||Y||²)
            Returns 1 if perfectly predictable, ~0 if independent
        """
        n_samples = X.shape[0]
        
        # Center the data
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # Scale to unit variance for numerical stability
        X_std = X.std() + 1e-10
        Y_std = Y.std() + 1e-10
        X_scaled = X / X_std
        Y_scaled = Y / Y_std
        
        # Compute optimal linear transformation using least squares
        # W = (X^T X)^{-1} X^T Y
        # Using lstsq for numerical stability (handles rank deficiency)
        W = np.linalg.lstsq(X_scaled, Y_scaled, rcond=None)[0]
        
        # Predict Y from X
        Y_pred = X_scaled @ W
        
        # Compute MSE (in scaled space)
        mse = np.mean((Y_scaled - Y_pred) ** 2)
        
        # Normalize by variance of Y (in scaled space)
        y_var = np.mean(Y_scaled ** 2)
        
        if y_var > 1e-10:
            normalized_error = mse / y_var
        else:
            normalized_error = 0.0
        
        # Convert error to similarity: 1 - error
        similarity = 1.0 - normalized_error
        
        # Clip to [0, 1]
        return np.clip(similarity, 0, 1)

    def invertible_affine_similarity(self, X: np.ndarray, Y: np.ndarray,
                                     epochs: int = 20, lr: float = 1e-2,
                                     reg_weight: float = 0.01,
                                     singular_floor: float = 1e-3) -> float:
        """
        Compute invertible-affine similarity: 1 - MSE between Y and Ŷ = XW + b
        with an invertibility regularizer on W.
        """
        device = self.device
        X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        Y_t = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        
        # Center and scale for consistency with L2/Procrustes
        X_t = X_t - X_t.mean(dim=0, keepdim=True)
        Y_t = Y_t - Y_t.mean(dim=0, keepdim=True)
        X_std = X_t.std() + 1e-10
        Y_std = Y_t.std() + 1e-10
        X_t = X_t / X_std
        Y_t = Y_t / Y_std
        
        out_features = Y_t.shape[1]
        in_features = X_t.shape[1]
        weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        bias = nn.Parameter(torch.zeros(out_features, device=device))
        
        if in_features == out_features:
            with torch.no_grad():
                weight.copy_(torch.eye(in_features, device=device))
        else:
            nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        optimizer = optim.Adam([weight, bias], lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = X_t @ weight.t() + bias
            loss = F.mse_loss(preds, Y_t)
            if reg_weight > 0:
                loss = loss + reg_weight * self._invertibility_regularizer(weight, singular_floor)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            preds = X_t @ weight.t() + bias
            mse = torch.mean((Y_t - preds) ** 2).item()
            var = torch.mean(Y_t ** 2).item()
        
        if var > 1e-10:
            similarity = 1.0 - (mse / var)
        else:
            similarity = 1.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def procrustes_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Procrustes similarity: 1 - MSE between Y and orthogonal transformation Ŷ = XQ.
        
        This measures how well Y can be predicted from X using an orthogonal transformation
        (rotation/reflection) that minimizes MSE. Higher similarity means representations differ
        only by rotation.
        
        For different dimensions, we project to the common subspace using the optimal
        orthogonal transformation that minimizes the Frobenius norm.
        """
        # Center the data
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # Scale to unit variance for numerical stability
        X_std = X.std() + 1e-10
        Y_std = Y.std() + 1e-10
        X_scaled = X / X_std
        Y_scaled = Y / Y_std
        
        d1, d2 = X_scaled.shape[1], Y_scaled.shape[1]
        
        # Compute optimal orthogonal transformation using Procrustes
        # For rectangular case: Q minimizes ||Y - XQ||_F
        # Solution: Q = U V^T where UΣV^T = SVD(X^T Y)
        M = X_scaled.T @ Y_scaled  # (d1, d2)
        U, S, Vt = svd(M, full_matrices=False)
        
        if d1 >= d2:
            Q = U[:, :d2] @ Vt
        else:
            Q = U @ Vt[:d1, :]
        
        # Predict Y from X using orthogonal transformation
        Y_pred = X_scaled @ Q
        
        # Compute MSE (in scaled space)
        mse = np.mean((Y_scaled - Y_pred) ** 2)
        
        # Normalize by variance of Y (in scaled space)
        y_var = np.mean(Y_scaled ** 2)
        
        if y_var > 1e-10:
            normalized_error = mse / y_var
        else:
            normalized_error = 0.0
        
        similarity = 1.0 - normalized_error
        return float(np.clip(similarity, 0.0, 1.0))
    
    def orthogonal_scaled_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Orthogonal + Isotropic Scaling similarity.
        
        This finds the optimal orthogonal matrix R and scalar s that minimize:
        ||Y - s*X*R||²
        
        This is invariant to:
        - Rotations and reflections (orthogonal transformations)
        - Uniform scaling (isotropic)
        
        More flexible than pure Procrustes (which has s=1), but more constrained
        than general affine transformation.
        """
        # Center and scale to unit variance for consistency with L2/Procrustes
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        X_std = X_centered.std() + 1e-10
        Y_std = Y_centered.std() + 1e-10
        X_centered = X_centered / X_std
        Y_centered = Y_centered / Y_std
        
        # Compute optimal orthogonal matrix R and scale s
        # Using Procrustes: minimize ||Y^T - s*R*X^T||_F
        M = Y_centered.T @ X_centered  # (features_Y, features_X)
        U, S_svd, Vt = svd(M, full_matrices=False)
        R = U @ Vt  # Optimal orthogonal matrix
        
        # Compute optimal isotropic scale s
        X_transformed_unscaled = X_centered @ R.T  # (samples, features_Y)
        numerator = np.sum(Y_centered * X_transformed_unscaled)  # Frobenius inner product
        denominator = np.sum(X_transformed_unscaled * X_transformed_unscaled)
        
        if denominator > 1e-10:
            s_opt = numerator / denominator
        else:
            s_opt = 1.0
        
        # Compute similarity: 1 - normalized error
        X_transformed = s_opt * X_transformed_unscaled
        error = np.linalg.norm(Y_centered - X_transformed, 'fro') ** 2
        norm_Y = np.linalg.norm(Y_centered, 'fro') ** 2
        
        if norm_Y > 1e-10:
            similarity = 1.0 - (error / norm_Y)
        else:
            similarity = 1.0  # Both are zero -> perfect match
        
        return float(np.clip(similarity, 0.0, 1.0))


def test_similarity_metrics():
    """Test the similarity metrics with synthetic data."""
    
    print("Testing Representational Similarity Metrics")
    print("=" * 80)
    
    # Create synthetic data
    n_samples = 100
    n_features1 = 256
    n_features2 = 512
    
    # Case 1: Identical representations (should give high similarity)
    X = np.random.randn(n_samples, n_features1)
    Y = X @ np.random.randn(n_features1, n_features2)  # Linear transformation
    
    sim = RepresentationalSimilarity()
    
    print("\nCase 1: Linearly related representations")
    results = sim.compute_all_metrics(torch.tensor(X), torch.tensor(Y))
    print(f"  CKA: {results['cka']:.4f}")
    print(f"  RSA: {results['rsa']:.4f}")
    print(f"  SVCCA: {results['svcca']:.4f}")
    
    # Case 2: Independent representations (should give low similarity)
    X = np.random.randn(n_samples, n_features1)
    Y = np.random.randn(n_samples, n_features2)
    
    print("\nCase 2: Independent representations")
    results = sim.compute_all_metrics(torch.tensor(X), torch.tensor(Y))
    print(f"  CKA: {results['cka']:.4f}")
    print(f"  RSA: {results['rsa']:.4f}")
    print(f"  SVCCA: {results['svcca']:.4f}")
    
    # Case 3: Same representation (should give ~1.0)
    X = np.random.randn(n_samples, n_features1)
    
    print("\nCase 3: Identical representations")
    results = sim.compute_all_metrics(torch.tensor(X), torch.tensor(X))
    print(f"  CKA: {results['cka']:.4f}")
    print(f"  RSA: {results['rsa']:.4f}")
    print(f"  SVCCA: {results['svcca']:.4f}")
    
    print("\n" + "=" * 80)
    print("Tests completed!")


if __name__ == '__main__':
    test_similarity_metrics()
