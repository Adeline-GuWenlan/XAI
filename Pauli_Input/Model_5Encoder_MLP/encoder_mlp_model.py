"""
Model 5: Encoder-MLP Architecture
A flexible model combining transformer encoder with MLP head.
Supports CLS token or weighted pooling methods.
Input: Pauli vectors of shape [B, N] where N is auto-detected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StructuredAttention(nn.Module):
    """Multi-head attention without physics mask, with optional CLS token support"""
    def __init__(self, d_model, nhead, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_cls_token = use_cls_token

        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape

        # Compute Q, K, V
        Q = self.q_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)

        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.nhead)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        output = self.out_linear(out)

        if return_attention:
            return output, attn_weights
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer without physics mask or dropout"""
    def __init__(self, d_model, nhead, dim_feedforward=512, use_cls_token=True):
        super().__init__()

        self.structured_attention = StructuredAttention(
            d_model, nhead, use_cls_token=use_cls_token
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network (no dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, return_attention=False):
        # Attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.structured_attention(x, return_attention=True)
        else:
            attn_out = self.structured_attention(x)
            attn_weights = None

        x = self.norm1(x + attn_out)

        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_weights
        return x


class PoolingLayer(nn.Module):
    """Flexible pooling layer supporting CLS token extraction or weighted aggregation"""
    def __init__(self, d_model, method='CLS'):
        """
        Args:
            d_model: Dimension of the model
            method: Pooling method - 'CLS' or 'weight'
                - 'CLS': Extract CLS token at position 0
                - 'weight': Weighted aggregation of all tokens
        """
        super().__init__()
        self.method = method.upper()
        self.d_model = d_model

        if self.method == 'WEIGHT':
            # Learnable weights for weighted aggregation
            self.weight_network = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, D]
               where N includes CLS token at position 0 if method='CLS'

        Returns:
            Pooled tensor of shape [B, D]
        """
        if self.method == 'CLS':
            # Extract CLS token at position 0
            return x[:, 0, :]  # [B, D]

        elif self.method == 'WEIGHT':
            # Compute attention weights for each token
            weights = self.weight_network(x)  # [B, N, 1]

            # Weighted sum across sequence dimension
            pooled = torch.sum(x * weights, dim=1)  # [B, D]

            return pooled

        else:
            raise ValueError(f"Unknown pooling method: {self.method}. Use 'CLS' or 'weight'.")


class MLPHead(nn.Module):
    """
    MLP head with ReLU activations
    Edit the structure directly in this class as needed
    """
    def __init__(self, input_dim):
        super().__init__()

        # Edit this structure as needed - default matches train_model4.py
        self.model = nn.Sequential(
            nn.Linear(input_dim,256),      # MLP layer 1
            nn.ReLU(),                      # ReLU activation 1
            nn.Linear(256, 128),            # MLP layer 2
            nn.ReLU(),                      # ReLU activation 2
            nn.Linear(128, 32),            # MLP layer 3
            nn.ReLU(),                      # ReLU activation 3
            nn.Linear(32, 1)               # Output layer
        )

    def forward(self, x):
        return self.model(x)


class EnsembleMLPHead(nn.Module):
    """
    Ensemble MLP head with voting mechanism using three MLPs of different depths
    Implements asymmetric ensemble similar to _build_asymmetric_ensemble
    """
    def __init__(self, input_dim, deep_weight=1.0, shallow_weight=1.0, medium_weight=1.0):
        """
        Args:
            input_dim: Input feature dimension
            deep_weight: Initial weight for deep pathway (default: 1.0)
            shallow_weight: Initial weight for shallow pathway (default: 1.0)
            medium_weight: Initial weight for medium pathway (default: 1.0)
        """
        super().__init__()

        # Deep pathway for complex patterns (5 layers)
        self.deep_pathway = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Shallow pathway for direct patterns (2 layers)
        self.shallow_pathway = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Medium pathway (3 layers)
        self.medium_pathway = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Learnable ensemble weights (initialized with provided values)
        initial_weights = torch.tensor([deep_weight, shallow_weight, medium_weight])
        self.ensemble_weights = nn.Parameter(initial_weights)

    def forward(self, x):
        """
        Forward pass with weighted ensemble voting

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Weighted combination of three pathway outputs [B, 1]
        """
        # Get outputs from different pathways
        deep_out = self.deep_pathway(x)
        shallow_out = self.shallow_pathway(x)
        medium_out = self.medium_pathway(x)

        # Apply softmax to weights for proper normalization
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted combination
        output = (weights[0] * deep_out +
                 weights[1] * shallow_out +
                 weights[2] * medium_out)

        return output


class LassoMLPHead(nn.Module):
    """
    MLP head with Lasso (L1) regularization
    The L1 penalty is applied during training via a custom loss calculation
    """
    def __init__(self, input_dim, lasso_lambda=0.01):
        """
        Args:
            input_dim: Input feature dimension
            lasso_lambda: L1 regularization penalty coefficient (default: 0.01)
        """
        super().__init__()
        self.lasso_lambda = lasso_lambda

        # MLP structure
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Predictions of shape [B, 1]
        """
        return self.model(x)

    def l1_penalty(self):
        """
        Compute L1 penalty on all model parameters

        Returns:
            L1 regularization term to be added to loss
        """
        l1_loss = 0.0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.lasso_lambda * l1_loss


class GBDTMLPHead(nn.Module):
    """
    Hybrid head combining Gradient Boosting Decision Tree with MLP
    Uses sklearn's GradientBoostingRegressor for tree-based learning
    followed by an MLP for final refinement

    Note: GBDT is fitted during training phase, not via backpropagation
    """
    def __init__(
        self,
        input_dim,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        learning_rate=0.1,
        subsample=1.0,
        use_mlp_refinement=True
    ):
        """
        Args:
            input_dim: Input feature dimension
            n_estimators: Number of boosting stages (trees) (default: 100)
            max_depth: Maximum depth of individual trees (default: 3)
            min_samples_split: Minimum samples required to split node (default: 2)
            min_samples_leaf: Minimum samples required at leaf node (default: 1)
            max_leaf_nodes: Maximum number of leaf nodes (default: None, unlimited)
            learning_rate: Learning rate for boosting (default: 0.1)
            subsample: Fraction of samples for fitting trees (default: 1.0)
            use_mlp_refinement: Whether to use MLP for post-processing (default: True)
        """
        super().__init__()

        # Import here to avoid requiring sklearn if not using this head
        from sklearn.ensemble import GradientBoostingRegressor

        self.input_dim = input_dim
        self.use_mlp_refinement = use_mlp_refinement

        # GBDT parameters (stored for replicability)
        self.gbdt_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_leaf_nodes': max_leaf_nodes,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'random_state': 42
        }

        # Initialize GBDT model (will be fitted during training)
        self.gbdt = GradientBoostingRegressor(**self.gbdt_params)
        self.gbdt_fitted = False

        # Optional MLP refinement layer
        if use_mlp_refinement:
            self.mlp_refinement = nn.Sequential(
                nn.Linear(1, 32),  # GBDT output (1) -> hidden layer
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)  # Final output
            )

    def fit_gbdt(self, X, y):
        """
        Fit the GBDT model on training data
        Should be called before forward pass during inference

        Args:
            X: Training features, numpy array or tensor of shape [N, input_dim]
            y: Training targets, numpy array or tensor of shape [N,] or [N, 1]
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        # Flatten y if needed
        if len(y.shape) > 1:
            y = y.flatten()

        # Fit GBDT
        self.gbdt.fit(X, y)
        self.gbdt_fitted = True

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Predictions of shape [B, 1]
        """
        # Convert to numpy for GBDT prediction
        x_np = x.cpu().detach().numpy()

        # Get GBDT predictions
        if not self.gbdt_fitted:
            # If GBDT not fitted, return zeros (should fit before using)
            gbdt_pred = torch.zeros(x.shape[0], 1, device=x.device)
        else:
            gbdt_pred = self.gbdt.predict(x_np)
            gbdt_pred = torch.from_numpy(gbdt_pred).float().to(x.device).unsqueeze(1)

        # Apply MLP refinement if enabled
        if self.use_mlp_refinement:
            output = self.mlp_refinement(gbdt_pred)
        else:
            output = gbdt_pred

        return output

    def get_feature_importance(self):
        """
        Get feature importance from the fitted GBDT

        Returns:
            Feature importance array or None if not fitted
        """
        if self.gbdt_fitted:
            return self.gbdt.feature_importances_
        return None

class EncoderMLPModel(nn.Module):
    """
    Complete Encoder-MLP model architecture for Pauli vectors

    Architecture:
        Input [B, N] Pauli vectors (N is auto-detected)
        → Reshape to [B, N, 1] (N tokens, 1 feature each)
        → Project to [B, N, d_model]
        → Optional CLS token → [B, N+1, d_model] or [B, N, d_model]
        → Transformer encoder layers
        → Pooling (CLS extraction or weighted aggregation) → [B, d_model]
        → MLP head → [B, 1]
    """
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        pooling_method='CLS',
        head_type='mlp',
        head_params=None
    ):
        """
        Args:
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            pooling_method: 'CLS' or 'weight'
            head_type: Type of regression head - 'mlp', 'ensemble', 'lasso', or 'gbdt'
            head_params: Dictionary of parameters for the regression head (optional)
        """
        super().__init__()

        self.d_model = d_model
        self.pooling_method = pooling_method.upper()
        self.use_cls_token = (self.pooling_method == 'CLS')
        self.head_type = head_type.lower()
        self.head_params = head_params if head_params is not None else {}

        # Input projection: each Pauli coefficient (1 feature) → d_model
        self.input_projection = nn.Linear(1, d_model)

        # CLS token (learnable) - only if using CLS pooling method
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                use_cls_token=self.use_cls_token
            )
            for _ in range(num_encoder_layers)
        ])

        # Pooling layer
        self.pooling = PoolingLayer(d_model=d_model, method=pooling_method)

        # MLP head - select based on head_type
        if self.head_type == 'mlp':
            self.mlp_head = MLPHead(input_dim=d_model)
        elif self.head_type == 'ensemble':
            self.mlp_head = EnsembleMLPHead(input_dim=d_model, **self.head_params)
        elif self.head_type == 'lasso':
            self.mlp_head = LassoMLPHead(input_dim=d_model, **self.head_params)
        elif self.head_type == 'gbdt':
            self.mlp_head = GBDTMLPHead(input_dim=d_model, **self.head_params)
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}. Use 'mlp', 'ensemble', 'lasso', or 'gbdt'.")

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor of shape [B, N] - Pauli vectors (N auto-detected)
            return_attention: Whether to return attention weights

        Returns:
            output: Predictions of shape [B, 1]
            attention_weights: (optional) List of attention weights from each layer
        """
        B, N = x.shape

        # Reshape Pauli vector [B, N] to [B, N, 1]
        # Each of the N Pauli coefficients becomes a token with 1 feature
        x = x.unsqueeze(-1)  # [B, N, 1]

        # Project each token from 1 feature to d_model dimensions
        x = self.input_projection(x)  # [B, N, d_model]

        # Prepend CLS token if using CLS pooling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, d_model]

        # Pass through encoder layers
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            if return_attention:
                x, attn_weights = encoder_layer(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = encoder_layer(x)

        # Pooling
        pooled = self.pooling(x)  # [B, d_model]

        # MLP head
        output = self.mlp_head(pooled)  # [B, 1]

        if return_attention:
            return output, attention_weights
        return output

    def get_pooled_representation(self, x):
        """
        Get the pooled representation before the MLP head
        Useful for visualization or transfer learning

        Args:
            x: Input tensor of shape [B, N] - Pauli vectors (N auto-detected)

        Returns:
            Pooled representation of shape [B, d_model]
        """
        B, N = x.shape

        # Reshape and project
        x = x.unsqueeze(-1)  # [B, N, 1]
        x = self.input_projection(x)  # [B, N, d_model]

        # Prepend CLS token if needed
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Pass through encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Pool
        pooled = self.pooling(x)

        return pooled


def create_encoder_mlp_model(
    pooling_method='CLS',
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    dim_feedforward=512,
    head_type='mlp',
    head_params=None
):
    """
    Factory function to create EncoderMLPModel with sensible defaults

    Example usage:
        # CLS token pooling with standard MLP head
        model = create_encoder_mlp_model(pooling_method='CLS', head_type='mlp')

        # Weighted aggregation pooling with ensemble head
        model = create_encoder_mlp_model(pooling_method='weight', head_type='ensemble')

        # Lasso head with custom penalty
        model = create_encoder_mlp_model(head_type='lasso', head_params={'lasso_lambda': 0.01})

        # GBDT head with custom tree parameters
        model = create_encoder_mlp_model(
            head_type='gbdt',
            head_params={'n_estimators': 100, 'max_depth': 3}
        )
    """
    return EncoderMLPModel(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        pooling_method=pooling_method,
        head_type=head_type,
        head_params=head_params
    )


if __name__ == "__main__":
    # Test the model with both pooling methods
    print("Testing EncoderMLPModel with Pauli vector input...")

    # Test dimensions - Pauli vectors
    batch_size = 4
    pauli_dim = 64  # 64 Pauli coefficients

    # Create dummy input - Pauli vectors
    x = torch.randn(batch_size, pauli_dim)

    print("\n1. Testing CLS token pooling:")
    model_cls = create_encoder_mlp_model(
        pooling_method='CLS',
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    output_cls = model_cls(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_cls.shape}")
    print(f"Output: {output_cls[:2]}")

    # Test with attention weights
    output_cls, attn_weights = model_cls(x, return_attention=True)
    print(f"Number of attention weight tensors: {len(attn_weights)}")
    print(f"Attention weights shape (first layer): {attn_weights[0].shape}")

    print("\n2. Testing weighted aggregation pooling:")
    model_weight = create_encoder_mlp_model(
        pooling_method='weight',
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model_weight.parameters()):,}")

    output_weight = model_weight(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_weight.shape}")
    print(f"Output: {output_weight[:2]}")

    # Test pooled representation extraction
    pooled_repr = model_weight.get_pooled_representation(x)
    print(f"\nPooled representation shape: {pooled_repr.shape}")

    print("\nAll tests passed!")
    print("\nNote: Edit the MLP structure directly in the MLPHead class as needed.")
