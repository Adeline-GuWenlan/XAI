"""
Model 5: Encoder-MLP Training Script
Training script for Encoder-MLP architecture with flexible pooling methods
Input: Pauli vectors of shape [B, 64]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from encoder_mlp_model import create_encoder_mlp_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class QuantumDataset(Dataset):
    """Dataset for Pauli vectors and SN labels"""
    def __init__(self, X_path, Y_path):
        self.X = torch.from_numpy(np.load(X_path)).float()
        # Reshape Y to [n, 1] to match model output shape [batch_size, 1]
        self.Y = torch.from_numpy(np.load(Y_path)).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model with comprehensive metrics"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets for correlation metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute metrics
    mse = total_loss / num_batches

    # R-squared
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(all_predictions, all_targets)

    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(all_predictions, all_targets)

    metrics = {
        'mse': mse,
        'r_squared': r_squared,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
        'pearson': pearson_corr,
        'pearson_pval': pearson_pval
    }

    return metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Encoder-MLP model')
    parser.add_argument('--pooling', type=str, default='CLS', choices=['CLS', 'weight'],
                        help='Pooling method: CLS or weight')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Dimension of transformer model')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Dimension of feedforward network')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.95,
                        help='Train/validation split ratio')

    args = parser.parse_args()

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"./results_{args.pooling.lower()}_{timestamp}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Data paths
    DATA_DIR = "/Users/guwenlan/Desktop/XAI/Full_Ori/3q"
    X_PATH = os.path.join(DATA_DIR, "3q_100000samples_uniform_20251114_182049_pauli_vectors.npy")
    Y_PATH = os.path.join(DATA_DIR, "3q_100000samples_uniform_20251114_182049_sn_labels.npy")

    # Device - prioritize MPS (Apple Silicon GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load dataset
    print("Loading dataset...")
    dataset = QuantumDataset(X_PATH, Y_PATH)

    # Split into train and validation
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    print(f"Input shape: {dataset.X.shape}")

    # Create model
    print(f"\nCreating Encoder-MLP model with {args.pooling.upper()} pooling...")
    model = create_encoder_mlp_model(
        pooling_method=args.pooling,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    # Open epoch log file
    epoch_log_path = os.path.join(RESULTS_DIR, "epoch_log.txt")
    with open(epoch_log_path, 'w') as epoch_log:
        epoch_log.write("Epoch\tTrain_Loss\tVal_MSE\tVal_R2\tVal_Spearman\tVal_Pearson\n")

        train_losses = []
        val_metrics_history = {
            'mse': [],
            'r_squared': [],
            'spearman': [],
            'pearson': []
        }
        best_val_loss = float('inf')

        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_metrics_history['mse'].append(val_metrics['mse'])
            val_metrics_history['r_squared'].append(val_metrics['r_squared'])
            val_metrics_history['spearman'].append(val_metrics['spearman'])
            val_metrics_history['pearson'].append(val_metrics['pearson'])

            # Log each epoch to file
            epoch_log.write(f"{epoch+1}\t{train_loss:.6f}\t{val_metrics['mse']:.6f}\t"
                          f"{val_metrics['r_squared']:.6f}\t{val_metrics['spearman']:.6f}\t"
                          f"{val_metrics['pearson']:.6f}\n")
            epoch_log.flush()

            # Print to console
            print(f"Epoch {epoch+1:3d}/{args.num_epochs}: Train Loss = {train_loss:.6f}, "
                  f"Val MSE = {val_metrics['mse']:.6f}, R² = {val_metrics['r_squared']:.4f}, "
                  f"Spearman = {val_metrics['spearman']:.4f}, Pearson = {val_metrics['pearson']:.4f}")

            # Save best model
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    print("=" * 80)
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Epoch log saved to: {epoch_log_path}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: MSE Loss
    axes[0, 0].plot(train_losses, label='Train MSE', color='blue')
    axes[0, 0].plot(val_metrics_history['mse'], label='Val MSE', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('MSE Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: R-squared
    axes[0, 1].plot(val_metrics_history['r_squared'], label='R²', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('R² Score Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Spearman Correlation
    axes[1, 0].plot(val_metrics_history['spearman'], label='Spearman ρ', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Spearman Correlation')
    axes[1, 0].set_title('Spearman Rank Correlation Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Pearson Correlation
    axes[1, 1].plot(val_metrics_history['pearson'], label='Pearson r', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].set_title('Pearson Correlation Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'Model 5: Encoder-MLP ({args.pooling.upper()} pooling) - Training Metrics',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    # Save training log
    log = {
        "model": f"Encoder-MLP with {args.pooling.upper()} pooling",
        "timestamp": timestamp,
        "device": str(device),
        "hyperparameters": {
            "pooling_method": args.pooling,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "train_split": args.train_split
        },
        "data": {
            "X_path": X_PATH,
            "Y_path": Y_PATH,
            "input_shape": list(dataset.X.shape),
            "train_samples": train_size,
            "val_samples": val_size
        },
        "results": {
            "final_train_mse": float(train_losses[-1]),
            "final_val_mse": float(val_metrics_history['mse'][-1]),
            "final_val_r_squared": float(val_metrics_history['r_squared'][-1]),
            "final_val_spearman": float(val_metrics_history['spearman'][-1]),
            "final_val_pearson": float(val_metrics_history['pearson'][-1]),
            "best_val_mse": float(best_val_loss)
        }
    }

    with open(os.path.join(RESULTS_DIR, "training_log.json"), 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Training log saved to {os.path.join(RESULTS_DIR, 'training_log.json')}")

    # Save model parameters summary
    model_params = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_type": f"Encoder-MLP with {args.pooling.upper()} pooling",
        "architecture_details": {
            "input_shape": list(dataset.X.shape),
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "pooling_method": args.pooling,
            "use_cls_token": args.pooling.upper() == 'CLS',
            "mlp_structure": "64 -> 128 -> 256 -> 128 -> 1 (edit in MLPHead class)"
        }
    }

    with open(os.path.join(RESULTS_DIR, "model_parameters.json"), 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"Model parameters saved to {os.path.join(RESULTS_DIR, 'model_parameters.json')}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "final_model.pt"))
    print(f"Final model saved to {os.path.join(RESULTS_DIR, 'final_model.pt')}")

    # Create summary report
    summary = f"""
Model 5: Encoder-MLP with {args.pooling.upper()} Pooling - Training Summary
{'=' * 80}
Timestamp: {timestamp}
Device: {device}

Data:
- Input shape: {dataset.X.shape}
- Total samples: {len(dataset)}
- Training samples: {train_size}
- Validation samples: {val_size}

Hyperparameters:
- Pooling method: {args.pooling.upper()}
- Batch size: {args.batch_size}
- Learning rate: {args.learning_rate}
- Number of epochs: {args.num_epochs}
- Transformer d_model: {args.d_model}
- Number of attention heads: {args.nhead}
- Number of encoder layers: {args.num_encoder_layers}
- Feedforward dimension: {args.dim_feedforward}

Model Architecture:
- Type: Encoder-MLP with {args.pooling.upper()} pooling
- CLS token: {'Yes' if args.pooling.upper() == 'CLS' else 'No'}
- Input: [B, 64] Pauli vectors -> [B, 64, 1] tokens
- Projection: 1 -> {args.d_model}
- Encoder: {args.num_encoder_layers} layers
- Pooling: {args.pooling.upper()} method
- MLP: {args.d_model} -> 128 -> 256 -> 128 -> 1
- Total parameters: {model_params['total_parameters']:,}

Results:
- Final training MSE: {train_losses[-1]:.6f}
- Final validation MSE: {val_metrics_history['mse'][-1]:.6f}
- Best validation MSE: {best_val_loss:.6f}

Evaluation Metrics (Final Epoch):
- R² Score: {val_metrics_history['r_squared'][-1]:.6f}
- Spearman Correlation: {val_metrics_history['spearman'][-1]:.6f}
- Pearson Correlation: {val_metrics_history['pearson'][-1]:.6f}

Files saved:
- best_model.pt: Best model checkpoint
- final_model.pt: Final model after all epochs
- training_curves.png: Comprehensive metrics visualization
- training_log.json: Detailed training log
- model_parameters.json: Model parameters summary
- epoch_log.txt: Per-epoch training log

Note: Edit MLP structure in encoder_mlp_model.py -> MLPHead class
"""

    with open(os.path.join(RESULTS_DIR, "summary.txt"), 'w') as f:
        f.write(summary)
    print(summary)
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'summary.txt')}")


if __name__ == "__main__":
    main()
