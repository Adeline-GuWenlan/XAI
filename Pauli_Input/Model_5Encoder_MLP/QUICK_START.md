# Quick Start - Encoder-MLP Training

## Ready to Train! ✓

All dependencies are installed and data files are verified.

## Fastest Way to Start

### 1. Edit Configuration (1 minute)
```bash
nano config.yaml
```
Change what you want to tune:
- Line 33: `batch_size`
- Line 34: `learning_rate`
- Line 35: `num_epochs`
- Line 44: `d_model`
- Line 45: `nhead`
- Line 46: `num_encoder_layers`
- Line 51: `pooling_method` (CLS or weight)
- Line 57: `type` (mlp, ensemble, lasso, or gbdt)

### 2. Run Training
```bash
cd /Users/guwenlan/Desktop/XAI/Pauli_Input/Model_5Encoder_MLP
python train_with_config.py --config config.yaml
```

OR use command line directly:
```bash
python train_encoder_mlp.py --pooling CLS --head_type mlp
```

## What You Can Tune

### Core Parameters (Highest Impact)
| Parameter | Current | Try These | Impact |
|-----------|---------|-----------|---------|
| `d_model` | 64 | 32, 64, 128, 256 | Model capacity |
| `num_encoder_layers` | 2 | 1, 2, 3, 4 | Model depth |
| `learning_rate` | 0.001 | 0.0001-0.005 | Convergence |
| `head_type` | mlp | mlp, ensemble, lasso, gbdt | Architecture |

### Pooling Methods
- **CLS**: CLS token extraction (like BERT)
- **weight**: Weighted aggregation (learnable attention)

### Regression Head Types

| Type | Best For | Key Parameter |
|------|----------|---------------|
| **mlp** | General purpose | - |
| **ensemble** | Multiple pattern types | `deep_weight`, `shallow_weight`, `medium_weight` |
| **lasso** | Sparse models, interpretability | `lasso_lambda` (0.001-1.0) |
| **gbdt** | Tree-based + neural hybrid | `n_estimators`, `max_depth` |

## Example Commands

### Quick Test (Fast)
```bash
python train_encoder_mlp.py --d_model 32 --num_encoder_layers 1 \
    --batch_size 64 --num_epochs 10
```

### Standard Training (Default)
```bash
python train_encoder_mlp.py --pooling CLS --head_type mlp
```

### High Capacity Model
```bash
python train_encoder_mlp.py --d_model 128 --nhead 8 \
    --num_encoder_layers 4 --batch_size 16 --num_epochs 50
```

### Try Different Heads
```bash
# Ensemble
python train_encoder_mlp.py --head_type ensemble

# Lasso (with regularization)
python train_encoder_mlp.py --head_type lasso

# GBDT (requires scikit-learn)
python train_encoder_mlp.py --head_type gbdt
```

## Output

Results saved to: `./results_<pooling>_<head>_<timestamp>/`

Files created:
- `best_model.pt` - Best checkpoint
- `training_curves.png` - All metrics visualization
- `training_log.json` - Full configuration + results
- `model_parameters.json` - Detailed parameter breakdown
- `summary.txt` - Human-readable summary

## Metrics Tracked

1. **MSE**: Mean squared error (lower is better)
2. **R²**: Explained variance (0-1, higher is better)
3. **Spearman ρ**: Rank correlation (higher is better)
4. **Pearson r**: Linear correlation (higher is better)

## Files Available

```
Model_5Encoder_MLP/
├── encoder_mlp_model.py           # Model architecture
├── train_encoder_mlp.py           # Training script (direct)
├── train_with_config.py           # Training script (YAML config)
├── config.yaml                    # ← EDIT THIS for easy tuning
├── TRAINING_GUIDE.md              # Detailed documentation
├── QUICK_START.md                 # This file
└── regression_heads_documentation.md  # Head type details
```

## Need Help?

1. **Detailed guide**: Read `TRAINING_GUIDE.md`
2. **Head types explained**: Read `regression_heads_documentation.md`
3. **Parameter breakdown**: Check `config.yaml` comments
4. **Model architecture**: Review `encoder_mlp_model.py`

## Next Steps

1. ✅ Run with default settings first
2. ✅ Compare different head types
3. ✅ Tune d_model and num_encoder_layers
4. ✅ Experiment with learning rates
5. ✅ Try both pooling methods

Happy training! 🚀
