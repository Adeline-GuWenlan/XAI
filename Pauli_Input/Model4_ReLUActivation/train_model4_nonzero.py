"""
Model 4: MLP with ReLU + ReLU - Non-zero Labels Only
Training script for regression on filtered dataset (label > 0 only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = "/Users/guwenlan/Desktop/XAI/Full_Ori/2q"

# Dataset paths
X_PATH = os.path.join(DATA_DIR, "2q_100000samples_uniform_20251114_194912_pauli_vectors.npy")
Y_PATH = os.path.join(DATA_DIR, "2q_100000samples_uniform_20251114_194912_sn_labels.npy")


class QuantumDatasetNonZero(Dataset):
    """Dataset for Pauli vectors and SN labels - FILTERED to include only label > 0"""
    def __init__(self, X_path, Y_path):
        # Load full datasets
        X_full = np.load(X_path)
        Y_full = np.load(Y_path)

        # Filter: keep only samples where label > 0
        mask = Y_full > 0
        self.X = torch.from_numpy(X_full[mask]).float()
        # Reshape Y to [n, 1] to match model output shape [batch_size, 1]
        self.Y = torch.from_numpy(Y_full[mask]).float().unsqueeze(1)

        # Store filtering statistics
        self.original_size = len(Y_full)
        self.filtered_size = len(self.Y)
        self.removed_count = self.original_size - self.filtered_size
        self.label_min = float(np.min(Y_full[mask]))
        self.label_max = float(np.max(Y_full[mask]))
        self.label_mean = float(np.mean(Y_full[mask]))

        print(f"\nDataset Filtering Statistics:")
        print(f"  Original dataset size: {self.original_size}")
        print(f"  Filtered dataset size: {self.filtered_size}")
        print(f"  Removed samples (label==0): {self.removed_count}")
        print(f"  Retention rate: {100*self.filtered_size/self.original_size:.2f}%")
        print(f"  Label range: [{self.label_min:.6f}, {self.label_max:.6f}]")
        print(f"  Label mean: {self.label_mean:.6f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLPReLUActivation(nn.Module):
    """MLP with ReLU + ReLU"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2, output_dim=1):
        super(MLPReLUActivation, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # MLP layer 1
            nn.ReLU(),                              # ReLU activation 1
            nn.Linear(hidden_dim, hidden_dim),      # MLP layer 2
            nn.ReLU(),                              # ReLU activation 2
            nn.Dropout(dropout_rate),               # Dropout
            nn.Linear(hidden_dim, output_dim)       # Output layer
        )

    def forward(self, x):
        return self.model(x)


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
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"./results_nonzero_{timestamp}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    TRAIN_SPLIT = 0.95
    HIDDEN_DIM = 64
    DROPOUT_RATE = 0.0

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

    # Load dataset (automatically filters to label > 0)
    print("Loading and filtering dataset...")
    dataset = QuantumDatasetNonZero(X_PATH, Y_PATH)

    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create model
    input_dim = dataset.X.shape[1]
    model = MLPReLUActivation(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        for epoch in range(NUM_EPOCHS):
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
            epoch_log.flush()  # Ensure it's written immediately

            # Print to console
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: Train Loss = {train_loss:.6f}, "
                  f"Val MSE = {val_metrics['mse']:.6f}, R² = {val_metrics['r_squared']:.4f}, "
                  f"Spearman = {val_metrics['spearman']:.4f}, Pearson = {val_metrics['pearson']:.4f}")

            # Save best model
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    print("=" * 80)
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Epoch log saved to: {epoch_log_path}")

    # Plot training curves with multiple subplots
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

    plt.suptitle('Model 4: MLP with ReLU + ReLU - Training Metrics (Non-zero Labels Only)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    # Save training log
    log = {
        "model": "MLP with ReLU + ReLU - Non-zero Labels Only",
        "task": "Regression on filtered dataset (label > 0 only)",
        "timestamp": timestamp,
        "device": str(device),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "train_split": TRAIN_SPLIT,
            "hidden_dim": HIDDEN_DIM,
            "dropout_rate": DROPOUT_RATE
        },
        "data": {
            "X_path": X_PATH,
            "Y_path": Y_PATH,
            "input_dim": input_dim,
            "original_dataset_size": dataset.original_size,
            "filtered_dataset_size": dataset.filtered_size,
            "removed_samples": dataset.removed_count,
            "retention_rate": dataset.filtered_size / dataset.original_size,
            "label_range": [dataset.label_min, dataset.label_max],
            "label_mean": dataset.label_mean,
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
        "model_architecture": str(model),
        "layer_details": {
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": 1,
            "activation_1": "ReLU",
            "activation_2": "ReLU",
            "dropout_rate": DROPOUT_RATE
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
Model 4: MLP with ReLU + ReLU - Training Summary (Non-zero Labels Only)
========================================================================
Task: Regression on filtered dataset (label > 0 only)
Timestamp: {timestamp}
Device: {device}

Data Filtering:
- Original dataset size: {dataset.original_size}
- Filtered dataset size: {dataset.filtered_size}
- Removed samples (label==0): {dataset.removed_count}
- Retention rate: {100*dataset.filtered_size/dataset.original_size:.2f}%
- Label range: [{dataset.label_min:.6f}, {dataset.label_max:.6f}]
- Label mean: {dataset.label_mean:.6f}

Data Split:
- Input dimension: {input_dim}
- Training samples: {train_size}
- Validation samples: {val_size}

Hyperparameters:
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}
- Number of epochs: {NUM_EPOCHS}
- Hidden dimension: {HIDDEN_DIM}
- Dropout rate: {DROPOUT_RATE}

Model Architecture:
- Type: MLP with Sequential layers
- Layer 1: Linear({input_dim} -> {HIDDEN_DIM})
- Activation 1: ReLU
- Layer 2: Linear({HIDDEN_DIM} -> {HIDDEN_DIM})
- Activation 2: ReLU
- Dropout: {DROPOUT_RATE}
- Output: Linear({HIDDEN_DIM} -> 1)
- Total parameters: {model_params['total_parameters']}

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
"""

    with open(os.path.join(RESULTS_DIR, "summary.txt"), 'w') as f:
        f.write(summary)
    print(summary)
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'summary.txt')}")


if __name__ == "__main__":
    main()
