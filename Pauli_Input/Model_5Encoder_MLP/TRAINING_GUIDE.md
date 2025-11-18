# Encoder-MLP Model Training Guide

## Quick Start

### Method 1: Using YAML Configuration (Recommended)

1. **Edit configuration** in `config.yaml`:
   ```bash
   nano config.yaml  # or use any text editor
   ```

2. **Run training with config**:
   ```bash
   python train_with_config.py --config config.yaml
   ```

3. **Dry run (preview without training)**:
   ```bash
   python train_with_config.py --config config.yaml --dry-run
   ```

### Method 2: Using Command Line Arguments

Run training directly with arguments:

```bash
# Basic example with MLP head
python train_encoder_mlp.py --pooling CLS --head_type mlp \
    --d_model 64 --nhead 4 --num_encoder_layers 2

# Ensemble head example
python train_encoder_mlp.py --pooling CLS --head_type ensemble \
    --d_model 128 --nhead 8 --batch_size 16 --learning_rate 0.0005

# Lasso head example
python train_encoder_mlp.py --pooling weight --head_type lasso \
    --num_epochs 50

# GBDT head example (requires scikit-learn)
python train_encoder_mlp.py --pooling CLS --head_type gbdt \
    --d_model 64 --num_encoder_layers 3
```

## Prerequisites

### Required Dependencies
```bash
pip install torch numpy matplotlib scipy
```

### Optional Dependencies
```bash
# For YAML configuration support
pip install pyyaml

# For GBDT head
pip install scikit-learn
```

## Configuration File Structure

The `config.yaml` file is organized into sections:

### 1. **Data Configuration**
- `data_dir`: Path to dataset directory
- `X_filename`: Input features file (Pauli vectors)
- `Y_filename`: Labels file (SN values)
- `train_split`: Train/validation split ratio

### 2. **Training Hyperparameters**
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimizer
- `num_epochs`: Number of training epochs
- `optimizer`: Optimizer type (adam, sgd, adamw)

### 3. **Model Architecture**
- `d_model`: Transformer embedding dimension
- `nhead`: Number of attention heads
- `num_encoder_layers`: Number of transformer layers
- `dim_feedforward`: Feedforward network dimension
- `pooling_method`: CLS token or weighted aggregation

### 4. **Regression Head Configuration**
Four types of regression heads available:

#### **MLP Head** (Standard)
```yaml
regression_head:
  type: "mlp"
```
- Simple feedforward network: d_model → 128 → 256 → 128 → 1
- Edit structure in `encoder_mlp_model.py` → `MLPHead` class

#### **Ensemble Head** (Multiple Pathways)
```yaml
regression_head:
  type: "ensemble"
  ensemble:
    deep_weight: 1.0      # 5-layer pathway
    shallow_weight: 1.0   # 2-layer pathway
    medium_weight: 1.0    # 3-layer pathway
```
- Three parallel pathways with learnable weights
- Good for capturing patterns at different complexities

#### **Lasso Head** (L1 Regularization)
```yaml
regression_head:
  type: "lasso"
  lasso:
    lasso_lambda: 0.01    # L1 penalty coefficient
```
- Promotes sparsity in weights
- Good for feature selection and interpretability

#### **GBDT Head** (Gradient Boosting + MLP)
```yaml
regression_head:
  type: "gbdt"
  gbdt:
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.1
    use_mlp_refinement: true
```
- Hybrid approach: tree-based + neural network
- Requires `scikit-learn`

## Parameter Tuning Guide

### Start Here (Highest Impact Parameters)

1. **d_model** (32, 64, 128, 256)
   - Controls model capacity
   - Larger = more capacity but slower training
   - Must be divisible by `nhead`

2. **num_encoder_layers** (1, 2, 3, 4)
   - Depth of transformer
   - More layers = deeper feature extraction

3. **learning_rate** (0.0001 to 0.005)
   - Critical for convergence
   - Larger models typically need lower LR

4. **head_type** (mlp, ensemble, lasso, gbdt)
   - Different inductive biases
   - Try multiple types for comparison

### Pooling Method Selection

- **CLS**: Uses a learnable CLS token (similar to BERT)
  - Good for classification-like tasks
  - Adds one extra parameter: cls_token

- **weight**: Weighted aggregation of all tokens
  - Attention over all Pauli coefficients
  - More parameters in pooling layer

### Head-Specific Tuning

#### Ensemble Head
- Start with equal weights (1.0, 1.0, 1.0)
- Increase `shallow_weight` if patterns are simple
- Increase `deep_weight` if patterns are complex
- Weights become learnable during training

#### Lasso Head
- Higher `lasso_lambda` (0.1, 1.0) → More sparsity
- Lower `lasso_lambda` (0.001, 0.01) → More capacity
- Monitor parameter magnitudes during training

#### GBDT Head
- `n_estimators` × `learning_rate` ≈ constant
  - More estimators → Lower learning rate
- `max_depth`: Start with 3, increase if underfitting
- `subsample` < 1.0 adds regularization (try 0.8)

## Common Workflows

### Workflow 1: Baseline Comparison
Compare all four head types with default settings:

```bash
# MLP baseline
python train_encoder_mlp.py --head_type mlp

# Ensemble comparison
python train_encoder_mlp.py --head_type ensemble

# Lasso comparison
python train_encoder_mlp.py --head_type lasso

# GBDT comparison
python train_encoder_mlp.py --head_type gbdt
```

### Workflow 2: Architecture Search
Test different model capacities:

```bash
# Small model
python train_encoder_mlp.py --d_model 32 --nhead 4 --num_encoder_layers 1

# Medium model (default)
python train_encoder_mlp.py --d_model 64 --nhead 4 --num_encoder_layers 2

# Large model
python train_encoder_mlp.py --d_model 128 --nhead 8 --num_encoder_layers 4
```

### Workflow 3: Pooling Method Comparison

```bash
# CLS token pooling
python train_encoder_mlp.py --pooling CLS

# Weighted aggregation pooling
python train_encoder_mlp.py --pooling weight
```

### Workflow 4: Using Configuration Presets

Edit `config.yaml` and uncomment one of the presets, then:

```bash
python train_with_config.py --config config.yaml
```

## Output Files

Training creates a timestamped results directory with:

- **best_model.pt**: Best model checkpoint (lowest validation loss)
- **final_model.pt**: Final model after all epochs
- **training_curves.png**: Visualization of all metrics
- **training_log.json**: Complete training configuration and results
- **model_parameters.json**: Detailed parameter breakdown
- **epoch_log.txt**: Per-epoch metrics (tab-separated)
- **summary.txt**: Human-readable training summary

## Evaluation Metrics

The training script tracks multiple metrics:

1. **MSE** (Mean Squared Error): Primary loss function
2. **R²** (R-squared): Coefficient of determination (0 to 1, higher is better)
3. **Spearman ρ**: Rank correlation (-1 to 1, higher is better)
4. **Pearson r**: Linear correlation (-1 to 1, higher is better)

## Tips and Best Practices

### 1. Start Simple
- Begin with small `d_model` and few `num_encoder_layers`
- Use default MLP head first
- Only increase complexity if underfitting

### 2. Monitor Overfitting
- Watch train vs validation MSE
- If gap is large: reduce capacity, add regularization
- Try Lasso head for built-in regularization

### 3. Learning Rate Tuning
- If loss diverges: reduce learning rate
- If loss plateaus early: increase learning rate
- Try learning rate warmup for large models

### 4. Batch Size Effects
- Larger batch: faster training, more stable gradients
- Smaller batch: more noise, potential regularization
- If out of memory: reduce batch_size or d_model

### 5. Use Multiple Metrics
- MSE for primary objective
- R² for proportion of variance explained
- Spearman for monotonic relationships
- Compare across all metrics

### 6. Experiment Tracking
- Use meaningful experiment names in config.yaml
- Add notes and tags for organization
- Keep config files for reproducibility

## Troubleshooting

### Issue: "Out of memory"
**Solution**: Reduce `batch_size`, `d_model`, or `num_encoder_layers`

### Issue: "ModuleNotFoundError: No module named 'yaml'"
**Solution**: `pip install pyyaml`

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: `pip install scikit-learn` (only needed for GBDT head)

### Issue: Training loss not decreasing
**Solution**:
- Check learning rate (try 0.001, 0.0001)
- Verify data is loaded correctly
- Increase model capacity (d_model, num_encoder_layers)

### Issue: Validation loss much worse than training loss
**Solution**:
- Model is overfitting
- Reduce capacity or add regularization
- Try Lasso head with higher lambda

### Issue: "d_model must be divisible by nhead"
**Solution**: Ensure d_model % nhead == 0
- Valid: d_model=64, nhead=4 ✓
- Invalid: d_model=64, nhead=5 ✗

## Advanced Usage

### Custom Head Structures

To create custom MLP architectures, edit `encoder_mlp_model.py`:

```python
class MLPHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),   # Customize these layers
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

### Hyperparameter Search

Use the `search_spaces` section in `config.yaml` as a guide for systematic hyperparameter optimization.

## Questions?

- Check `regression_heads_documentation.md` for detailed head descriptions
- Review `encoder_mlp_model.py` for architecture implementation
- Review `train_encoder_mlp.py` for training loop details
