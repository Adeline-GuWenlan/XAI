# Model 5: Encoder-MLP Architecture

A flexible transformer-based model combining encoder layers with MLP head for quantum state prediction.

## Architecture Overview

```
Input → Projection → [Optional CLS Token] → Encoder Layers → Pooling → MLP Head → Output
```

### Components

1. **Input Projection**: Projects input features to transformer dimension (d_model)
2. **CLS Token** (optional): Learnable token prepended to sequence when using CLS pooling
3. **Encoder Layers**: Transformer encoder without physics mask, supports full attention
4. **Pooling Layer**: Flexible pooling supporting two methods:
   - **CLS**: Extracts CLS token representation
   - **Weight**: Learnable weighted aggregation across all tokens
5. **MLP Head**: Multi-layer perceptron with ReLU activations for final prediction

## Key Features

- **No Physics Mask**: Full attention between all tokens
- **Flexible Pooling**: Choose between CLS token extraction or weighted aggregation
- **Modular Design**: Easy to customize each component
- **Comprehensive Evaluation**: Includes MSE, R², Spearman, and Pearson metrics

## Files

- `encoder_mlp_model.py`: Model definition and architecture
- `train_encoder_mlp.py`: Training script with full evaluation pipeline
- `README.md`: This documentation file

## Usage

### Basic Training

Train with CLS token pooling:
```bash
python train_encoder_mlp.py --pooling CLS
```

Train with weighted aggregation pooling:
```bash
python train_encoder_mlp.py --pooling weight
```

### Advanced Options

Full command with all available options:
```bash
python train_encoder_mlp.py \
    --pooling CLS \
    --d_model 64 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --dim_feedforward 512 \
    --dropout 0.05 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 30 \
    --train_split 0.95 \
    --mlp_hidden_dim1 128 \
    --mlp_hidden_dim2 256
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pooling` | str | 'CLS' | Pooling method: 'CLS' or 'weight' |
| `--d_model` | int | 64 | Transformer model dimension |
| `--nhead` | int | 4 | Number of attention heads |
| `--num_encoder_layers` | int | 2 | Number of encoder layers |
| `--dim_feedforward` | int | 512 | Feedforward network dimension |
| `--dropout` | float | 0.05 | Dropout rate |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 0.001 | Learning rate |
| `--num_epochs` | int | 30 | Number of training epochs |
| `--train_split` | float | 0.95 | Train/validation split ratio |
| `--mlp_hidden_dim1` | int | 128 | First MLP hidden dimension |
| `--mlp_hidden_dim2` | int | 256 | Second MLP hidden dimension |

## Using the Model in Code

### Create and Use Model

```python
from encoder_mlp_model import create_encoder_mlp_model
import torch

# Create model with CLS pooling
model_cls = create_encoder_mlp_model(
    input_dim=15,
    pooling_method='CLS',
    d_model=64,
    nhead=4,
    num_encoder_layers=2
)

# Create model with weighted pooling
model_weight = create_encoder_mlp_model(
    input_dim=15,
    pooling_method='weight',
    d_model=64,
    nhead=4,
    num_encoder_layers=2
)

# Forward pass
x = torch.randn(4, 16, 15)  # [batch_size, seq_len, input_dim]
output = model_cls(x)  # [batch_size, 1]

# Get attention weights
output, attn_weights = model_cls(x, return_attention=True)

# Get pooled representation (before MLP head)
pooled = model_cls.get_pooled_representation(x)  # [batch_size, d_model]
```

### Custom Model Configuration

```python
from encoder_mlp_model import EncoderMLPModel

# Fully customized model
model = EncoderMLPModel(
    input_dim=15,
    d_model=128,
    nhead=8,
    num_encoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    pooling_method='weight',
    mlp_hidden_dim1=256,
    mlp_hidden_dim2=512,
    output_dim=1
)
```

## Model Comparison: CLS vs Weight Pooling

### CLS Token Pooling
- **Pros**:
  - Standard transformer approach
  - Single global representation
  - Proven effective in many tasks
  - Clear interpretation
- **Cons**:
  - Adds extra learnable token
  - Slightly more parameters
  - Must learn to aggregate information into CLS token

### Weighted Aggregation Pooling
- **Pros**:
  - Learns attention over all tokens
  - No extra sequence length
  - Adaptive weighting
  - More flexible aggregation
- **Cons**:
  - Slightly more computation
  - Less standard approach
  - May be harder to interpret

## Output Files

After training, the following files are saved in `results_<pooling>_<timestamp>/`:

- `best_model.pt`: Model checkpoint with best validation MSE
- `final_model.pt`: Model after all training epochs
- `training_curves.png`: Visualization of all metrics over training
- `training_log.json`: Complete training configuration and results
- `model_parameters.json`: Model architecture details
- `epoch_log.txt`: Per-epoch metrics log
- `summary.txt`: Human-readable training summary

## Expected Input Format

The model expects input in the shape `[batch_size, sequence_length, input_dim]`:
- For flat Pauli vectors: The training script automatically reshapes `[B, input_dim]` to `[B, 1, input_dim]`
- For sequence data: Use natural `[B, N, input_dim]` format

## Architecture Details

### Encoder Layer
- Multi-head self-attention (no physics mask)
- Layer normalization
- Feedforward network with GELU activation
- Residual connections
- Dropout regularization

### MLP Head
```
Input (d_model)
  → Linear(d_model → hidden_dim1) → ReLU
  → Linear(hidden_dim1 → hidden_dim2) → ReLU
  → Linear(hidden_dim2 → hidden_dim1) → ReLU
  → Linear(hidden_dim1 → output_dim)
  → Output
```

## Performance Metrics

The training script tracks:
- **MSE**: Mean Squared Error (loss function)
- **R²**: Coefficient of determination
- **Spearman ρ**: Rank correlation coefficient
- **Pearson r**: Linear correlation coefficient

## Differences from Model 4

| Feature | Model 4 (Pure MLP) | Model 5 (Encoder-MLP) |
|---------|-------------------|----------------------|
| Architecture | Sequential MLP | Encoder + Pooling + MLP |
| Attention | None | Multi-head self-attention |
| Input format | Flat vector | Sequence of tokens |
| Pooling | N/A | CLS or weighted |
| Parameters | ~241K | ~241K (similar) |
| Complexity | Low | Medium |

## Tips for Best Results

1. **Start with CLS pooling** - It's the standard approach and often works well
2. **Try weighted pooling** if you have variable-length sequences or want adaptive aggregation
3. **Adjust d_model** based on your input complexity (64-128 works well for most cases)
4. **Increase num_encoder_layers** for more complex patterns (2-4 layers typical)
5. **Monitor all metrics** - Sometimes Spearman/Pearson correlation is more important than MSE

## Troubleshooting

**High training loss**: Try reducing learning rate or adding more dropout

**Overfitting**: Increase dropout, reduce model size, or add more training data

**Poor convergence**: Try different learning rate, check input normalization, or increase model capacity

**Memory issues**: Reduce batch_size, d_model, or num_encoder_layers

## Citation

If you use this model in your research, please cite:
```
Model 5: Encoder-MLP Architecture for Quantum State Prediction
```



Training completed! Best validation loss: 0.000008
Epoch log saved to: ./results_cls_mlp_20251118_121934/epoch_log.txt
Training curves saved to ./results_cls_mlp_20251118_121934/training_curves.png
Training log saved to ./results_cls_mlp_20251118_121934/training_log.json
Model parameters saved to ./results_cls_mlp_20251118_121934/model_parameters.json
Final model saved to ./results_cls_mlp_20251118_121934/final_model.pt

Model 5: Encoder-MLP with CLS Pooling and MLP Head - Training Summary
================================================================================
Timestamp: 20251118_121934
Device: cuda

Data:
- Input shape: torch.Size([100000, 16])
- Total samples: 100000
- Training samples: 95000
- Validation samples: 5000

Hyperparameters:
- Pooling method: CLS
- Regression head type: MLP
- Batch size: 32
- Learning rate: 0.0005
- Number of epochs: 30
- Transformer d_model: 256
- Number of attention heads: 4
- Number of encoder layers: 3
- Feedforward dimension: 512

Model Architecture:
- Type: Encoder-MLP with CLS pooling and MLP head
- CLS token: Yes
- Input: [B, 64] Pauli vectors -> [B, 64, 1] tokens
- Projection: 1 -> 256
- Encoder: 3 layers
- Pooling: CLS method
  - MLP: 256 -> 128 -> 256 -> 128 -> 1

Parameter Counts:
- Input projection + CLS: 768
- Encoder layers: 1,581,312
- Pooling layer: 0
- Regression head: 102,849
- Total parameters: 1,684,929

Results:
- Final training MSE: 0.000059
- Final validation MSE: 0.000008
- Best validation MSE: 0.000008

Evaluation Metrics (Final Epoch):
- R² Score: 0.999954
- Spearman Correlation: 0.999983
- Pearson Correlation: 0.999982

Files saved:
- best_model.pt: Best model checkpoint
- final_model.pt: Final model after all epochs
- training_curves.png: Comprehensive metrics visualization
- training_log.json: Detailed training log
- model_parameters.json: Model parameters summary
- epoch_log.txt: Per-epoch training log

Note: Edit regression head structures in encoder_mlp_model.py

Summary saved to ./results_cls_mlp_20251118_121934/summary.txt
(magic) [gwl@cdsw01 Model_5Encoder_MLP]$ 