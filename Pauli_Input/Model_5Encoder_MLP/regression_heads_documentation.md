# Regression Heads Documentation

This document provides detailed information about all regression head classes available in the Encoder-MLP model, including their parameters and usage.

---

## 1. MLPHead

**Description**: Standard MLP head with ReLU activations for regression tasks.

**Class**: `MLPHead(nn.Module)`

**Location**: `encoder_mlp_model.py`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Required | Input feature dimension (typically `d_model` from encoder) |

### Architecture

- **Layer 1**: Linear(`input_dim`, 128) + ReLU
- **Layer 2**: Linear(128, 256) + ReLU
- **Layer 3**: Linear(256, 128) + ReLU
- **Layer 4**: Linear(128, 1)

### Methods

#### `forward(x)`
- **Input**: `x` - Tensor of shape `[B, input_dim]`
- **Output**: Predictions of shape `[B, 1]`

### Usage Example

```python
from encoder_mlp_model import MLPHead

head = MLPHead(input_dim=64)
output = head(pooled_features)
```

---

## 2. EnsembleMLPHead

**Description**: Ensemble MLP head using voting mechanism with three MLPs of different depths. Implements asymmetric ensemble with learnable weights.

**Class**: `EnsembleMLPHead(nn.Module)`

**Location**: `encoder_mlp_model.py`

**Reference**: Based on `_build_asymmetric_ensemble` from `/Users/guwenlan/Desktop/XAI/Model_Archi/mlp.py`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Required | Input feature dimension |
| `deep_weight` | float | 1.0 | Initial weight for deep pathway |
| `shallow_weight` | float | 1.0 | Initial weight for shallow pathway |
| `medium_weight` | float | 1.0 | Initial weight for medium pathway |

### Architecture

The ensemble consists of three parallel pathways:

#### Deep Pathway (5 layers)
- Linear(`input_dim`, 256) + ReLU
- Linear(256, 128) + ReLU
- Linear(128, 64) + ReLU
- Linear(64, 32) + ReLU
- Linear(32, 1)

#### Shallow Pathway (2 layers)
- Linear(`input_dim`, 64) + ReLU
- Linear(64, 1)

#### Medium Pathway (3 layers)
- Linear(`input_dim`, 128) + ReLU
- Linear(128, 32) + ReLU
- Linear(32, 1)

### Learnable Parameters

- `ensemble_weights`: Learnable weights for combining pathways (initialized with provided values, then softmax-normalized)

### Methods

#### `forward(x)`
- **Input**: `x` - Tensor of shape `[B, input_dim]`
- **Output**: Weighted combination of three pathway outputs, shape `[B, 1]`
- **Computation**:
  ```
  weights = softmax(ensemble_weights)
  output = weights[0] * deep_out + weights[1] * shallow_out + weights[2] * medium_out
  ```

### Usage Example

```python
from encoder_mlp_model import EnsembleMLPHead

# Default equal weights
head = EnsembleMLPHead(input_dim=64)

# Custom initial weights
head = EnsembleMLPHead(
    input_dim=64,
    deep_weight=2.0,
    shallow_weight=1.0,
    medium_weight=1.5
)
```

### PyTorch Functions Used

- `torch.nn.Sequential`: Container for sequential layers
- `torch.nn.Linear`: Fully connected layer
- `torch.nn.ReLU`: ReLU activation
- `torch.nn.Parameter`: Learnable parameter wrapper
- `torch.nn.functional.softmax`: Softmax normalization

---

## 3. LassoMLPHead

**Description**: MLP head with Lasso (L1) regularization for sparse regression.

**Class**: `LassoMLPHead(nn.Module)`

**Location**: `encoder_mlp_model.py`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Required | Input feature dimension |
| `lasso_lambda` | float | 0.01 | L1 regularization penalty coefficient (λ) |

### Architecture

- **Layer 1**: Linear(`input_dim`, 128) + ReLU
- **Layer 2**: Linear(128, 256) + ReLU
- **Layer 3**: Linear(256, 128) + ReLU
- **Layer 4**: Linear(128, 1)

### Methods

#### `forward(x)`
- **Input**: `x` - Tensor of shape `[B, input_dim]`
- **Output**: Predictions of shape `[B, 1]`

#### `l1_penalty()`
- **Input**: None
- **Output**: L1 regularization term (scalar tensor)
- **Computation**: `λ * Σ|θ|` where θ represents all model parameters
- **Usage**: Add this to the loss during training:
  ```python
  loss = mse_loss + model.mlp_head.l1_penalty()
  ```

### Training Modifications Required

When using LassoMLPHead, the training loop must be modified to include the L1 penalty:

```python
predictions = model(X_batch)
base_loss = criterion(predictions, Y_batch)

# Add L1 penalty for Lasso regularization
if isinstance(model.mlp_head, LassoMLPHead):
    loss = base_loss + model.mlp_head.l1_penalty()
else:
    loss = base_loss
```

### Usage Example

```python
from encoder_mlp_model import LassoMLPHead

# Default lambda = 0.01
head = LassoMLPHead(input_dim=64)

# Custom lambda for stronger regularization
head = LassoMLPHead(input_dim=64, lasso_lambda=0.1)

# Weaker regularization
head = LassoMLPHead(input_dim=64, lasso_lambda=0.001)
```

### PyTorch Functions Used

- `torch.nn.Sequential`: Container for sequential layers
- `torch.nn.Linear`: Fully connected layer
- `torch.nn.ReLU`: ReLU activation
- `torch.sum`: Sum tensor elements
- `torch.abs`: Absolute value for L1 norm

---

## 4. GBDTMLPHead

**Description**: Hybrid head combining Gradient Boosting Decision Tree (GBDT) with optional MLP refinement for precise regression.

**Class**: `GBDTMLPHead(nn.Module)`

**Location**: `encoder_mlp_model.py`

**Dependencies**: `sklearn.ensemble.GradientBoostingRegressor`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Required | Input feature dimension |
| `n_estimators` | int | 100 | Number of boosting stages (trees) |
| `max_depth` | int | 3 | Maximum depth of individual regression trees |
| `min_samples_split` | int | 2 | Minimum number of samples required to split an internal node |
| `min_samples_leaf` | int | 1 | Minimum number of samples required to be at a leaf node |
| `max_leaf_nodes` | int or None | None | Maximum number of leaf nodes (None = unlimited) |
| `learning_rate` | float | 0.1 | Learning rate shrinks contribution of each tree |
| `subsample` | float | 1.0 | Fraction of samples used for fitting individual trees (0.0 < subsample ≤ 1.0) |
| `use_mlp_refinement` | bool | True | Whether to use MLP for post-processing GBDT output |

### Architecture

#### GBDT Component
- **Algorithm**: Gradient Boosting Decision Tree (sklearn)
- **Input**: Pooled features from encoder
- **Output**: Initial prediction

#### MLP Refinement (optional, if `use_mlp_refinement=True`)
- Linear(1, 32) + ReLU  # GBDT output → hidden layer
- Linear(32, 16) + ReLU
- Linear(16, 1)  # Final output

### Methods

#### `fit_gbdt(X, y)`
Fits the GBDT model on training data. **Must be called before using the model for inference.**

- **Input**:
  - `X`: Training features, numpy array or tensor of shape `[N, input_dim]`
  - `y`: Training targets, numpy array or tensor of shape `[N,]` or `[N, 1]`
- **Output**: None
- **Side effects**: Sets `self.gbdt_fitted = True`

#### `forward(x)`
- **Input**: `x` - Tensor of shape `[B, input_dim]`
- **Output**: Predictions of shape `[B, 1]`
- **Behavior**:
  - If GBDT not fitted, returns zeros
  - If fitted, returns GBDT predictions (optionally refined by MLP)

#### `get_feature_importance()`
- **Input**: None
- **Output**: Feature importance array (numpy) or None if not fitted
- **Description**: Returns feature importance scores from the fitted GBDT

### Training Workflow

The GBDTMLPHead requires a special training workflow:

1. **Extract pooled representations** from all training data
2. **Fit the GBDT** on these representations
3. **Train the MLP refinement** (if enabled) via backpropagation

```python
# Step 1: Extract pooled representations
model.eval()
all_pooled = []
all_targets = []
with torch.no_grad():
    for X_batch, Y_batch in train_loader:
        pooled = model.get_pooled_representation(X_batch.to(device))
        all_pooled.append(pooled.cpu())
        all_targets.append(Y_batch)

all_pooled = torch.cat(all_pooled, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Step 2: Fit GBDT
model.mlp_head.fit_gbdt(all_pooled, all_targets)

# Step 3: Train MLP refinement (if enabled) via normal training loop
# The GBDT is now frozen and only MLP refinement is trained
```

### Usage Example

```python
from encoder_mlp_model import GBDTMLPHead

# Default configuration
head = GBDTMLPHead(input_dim=64)

# Custom tree parameters for deeper trees
head = GBDTMLPHead(
    input_dim=64,
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    learning_rate=0.05
)

# GBDT only (no MLP refinement)
head = GBDTMLPHead(
    input_dim=64,
    use_mlp_refinement=False
)

# Constrained tree growth
head = GBDTMLPHead(
    input_dim=64,
    max_depth=4,
    max_leaf_nodes=16,
    min_samples_split=10
)
```

### Scikit-learn Functions Used

- `sklearn.ensemble.GradientBoostingRegressor`: GBDT regressor
  - Parameters passed: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `learning_rate`, `subsample`, `random_state`
  - Methods used: `fit(X, y)`, `predict(X)`, `feature_importances_`

### PyTorch Functions Used

- `torch.nn.Sequential`: Container for MLP refinement layers
- `torch.nn.Linear`: Fully connected layer
- `torch.nn.ReLU`: ReLU activation
- `torch.from_numpy`: Convert numpy array to tensor
- `torch.zeros`: Create zero tensor

---

## Summary Table

| Head Type | Parameters Count* | Special Features | Best For |
|-----------|------------------|------------------|----------|
| **MLPHead** | ~66K | Standard regression | General purpose |
| **EnsembleMLPHead** | ~143K | 3 pathways + learnable weights | Complex patterns with multiple scales |
| **LassoMLPHead** | ~66K | L1 regularization | Sparse feature selection |
| **GBDTMLPHead** | Varies** | Tree-based + optional MLP | Non-linear patterns, interpretability |

\* Approximate for `input_dim=64`
\** Depends on tree parameters and `use_mlp_refinement`

---

## Integration with EncoderMLPModel

All regression heads are integrated into the `EncoderMLPModel` class via the `head_type` parameter:

```python
from encoder_mlp_model import create_encoder_mlp_model

# Standard MLP head
model = create_encoder_mlp_model(head_type='mlp')

# Ensemble head with custom weights
model = create_encoder_mlp_model(
    head_type='ensemble',
    head_params={'deep_weight': 2.0, 'shallow_weight': 1.0, 'medium_weight': 1.5}
)

# Lasso head with custom penalty
model = create_encoder_mlp_model(
    head_type='lasso',
    head_params={'lasso_lambda': 0.01}
)

# GBDT head with custom tree parameters
model = create_encoder_mlp_model(
    head_type='gbdt',
    head_params={
        'n_estimators': 100,
        'max_depth': 3,
        'min_samples_split': 2,
        'use_mlp_refinement': True
    }
)
```

---

## Modification Guidelines

To modify regression head parameters in place (as per NOTE in encoder_mlp_model.py):

1. **Open**: `encoder_mlp_model.py`
2. **Locate**: The desired regression head class (`MLPHead`, `EnsembleMLPHead`, `LassoMLPHead`, or `GBDTMLPHead`)
3. **Edit**: Modify default parameter values in the `__init__` method
4. **Save**: The changes will apply to all new model instances

**Example**: To change LassoMLPHead default lambda from 0.01 to 0.1:
```python
# In encoder_mlp_model.py, line ~235
def __init__(self, input_dim, lasso_lambda=0.1):  # Changed from 0.01
    ...
```

---

## Notes

- All regression heads output shape `[B, 1]` for regression tasks
- Parameters are stored in model checkpoints for replicability
- GBDT head requires special training procedure (see above)
- Lasso head requires modified loss calculation (see above)
- Ensemble head learns optimal pathway weights during training

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Model File**: `encoder_mlp_model.py`
**Training Script**: `train_encoder_mlp.py`
