# Quick Start Guide for Model 5: Encoder-MLP

## Key Features

✓ **Auto-detects input length** - Works with any Pauli vector size (not hardcoded to 64)
✓ **No dropout** - Clean architecture for direct editing
✓ **Editable MLP** - Modify structure directly in `MLPHead` class
✓ **Flexible pooling** - Choose between CLS token or weighted aggregation
✓ **DataLoader included** - Training script handles batching and shuffling

## Quick Training

```bash
# Train with CLS token pooling (default)
python train_encoder_mlp.py --pooling CLS

# Train with weighted aggregation pooling
python train_encoder_mlp.py --pooling weight

# Custom settings
python train_encoder_mlp.py --pooling CLS --d_model 128 --num_encoder_layers 3 --learning_rate 0.0005
```

## Model Architecture

```
Input: [B, N] Pauli vectors (N auto-detected)
  ↓
Reshape to [B, N, 1] tokens
  ↓
Project: 1 → d_model
  ↓
[Optional] Add CLS token: [B, N+1, d_model]
  ↓
Encoder Layers (no dropout, no physics mask)
  ↓
Pooling:
  - CLS method: Extract token at position 0
  - Weight method: Learnable weighted sum
  ↓
Pooled: [B, d_model]
  ↓
MLP Head: d_model → 128 → 256 → 128 → 1
  ↓
Output: [B, 1]
```

## Editing MLP Structure

Open `encoder_mlp_model.py` and find the `MLPHead` class (around line 133):

```python
class MLPHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Edit this structure as needed
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

## DataLoader

The training script (`train_encoder_mlp.py`) includes a complete DataLoader setup:
- Loads Pauli vectors from `.npy` files
- Handles train/validation split
- Shuffles training data
- Batches data automatically

No need to define it elsewhere!

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pooling` | CLS | 'CLS' or 'weight' |
| `--d_model` | 64 | Transformer dimension |
| `--nhead` | 4 | Number of attention heads |
| `--num_encoder_layers` | 2 | Number of encoder layers |
| `--dim_feedforward` | 512 | FFN dimension |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--num_epochs` | 30 | Number of epochs |
| `--train_split` | 0.95 | Train/val split ratio |

## Files Structure

```
Model_5Encoder_MLP/
├── encoder_mlp_model.py     # Model definition
├── train_encoder_mlp.py     # Training script with DataLoader
├── README.md                # Detailed documentation
└── USAGE.md                 # This quick start guide
```

## Example Usage in Code

```python
from encoder_mlp_model import create_encoder_mlp_model
import torch

# Create model (auto-detects input length)
model = create_encoder_mlp_model(
    pooling_method='CLS',
    d_model=64,
    nhead=4,
    num_encoder_layers=2
)

# Input can be any length (e.g., 64, 128, 256, etc.)
x = torch.randn(4, 64)  # [batch_size, N]
output = model(x)       # [batch_size, 1]
```

## Output Files

After training, results are saved in `results_<pooling>_<timestamp>/`:
- `best_model.pt` - Best checkpoint
- `final_model.pt` - Final model
- `training_curves.png` - Metrics plots
- `training_log.json` - Complete log
- `epoch_log.txt` - Per-epoch metrics
- `summary.txt` - Human-readable summary
