"""
Compute R-squared (R²) metric for Model 2
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
import sys
import os

# Set random seeds for reproducibility (same as training)
torch.manual_seed(42)
np.random.seed(42)
# Results directory
RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "./results_20251112_145105"

print(f"Computing R² for results in: {RESULTS_DIR}")
print("=" * 80)

# Load training log to get configuration
with open(os.path.join(RESULTS_DIR, "training_log.json"), 'r') as f:
    log = json.load(f)

# Extract configuration
X_PATH = log["data"]["X_path"]
Y_PATH = log["data"]["Y_path"]
TRAIN_SPLIT = log["hyperparameters"]["train_split"]
HIDDEN_DIM = log["hyperparameters"]["hidden_dim"]
DROPOUT_RATE = log["hyperparameters"]["dropout_rate"]
INPUT_DIM = log["data"]["input_dim"]

print(f"Dataset: {os.path.basename(X_PATH)}")
print(f"Train/Val split: {TRAIN_SPLIT:.1%} / {1-TRAIN_SPLIT:.1%}")
print()


class QuantumDataset(Dataset):
    """Dataset for Pauli vectors and SN labels"""
    def __init__(self, X_path, Y_path):
        self.X = torch.from_numpy(np.load(X_path)).float()
        self.Y = torch.from_numpy(np.load(Y_path)).float()

        if len(self.Y.shape) == 1:
            self.Y = self.Y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLPSymmetricActivation(nn.Module):
    """MLP with Symmetric Activation (Tanh) + ReLU"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2, output_dim=1):
        super(MLPSymmetricActivation, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def compute_r_squared(y_true, y_pred):
    """
    Compute R-squared (coefficient of determination)
    R² = 1 - (SS_res / SS_tot)
    where:
        SS_res = sum of squared residuals = Σ(y_true - y_pred)²
        SS_tot = total sum of squares = Σ(y_true - mean(y_true))²
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def evaluate_with_r2(model, dataloader, device):
    """Evaluate model and compute R²"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch).cpu()

            all_predictions.append(predictions)
            all_targets.append(Y_batch)

    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    # Compute R²
    r2 = compute_r_squared(targets, predictions)

    # Also compute MSE for reference
    mse = torch.mean((predictions - targets) ** 2).item()

    return r2, mse, predictions, targets


# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print()

# Load dataset
print("Loading dataset...")
dataset = QuantumDataset(X_PATH, Y_PATH)

# Split into train and validation (same split as training)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Total samples: {len(dataset)}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")
print()

# Load model
print("Loading best model...")
model = MLPSymmetricActivation(INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt")))
model = model.to(device)
print("Model loaded successfully!")
print()

# Compute R² for training set
print("Computing R² for training set...")
train_r2, train_mse, train_pred, train_target = evaluate_with_r2(model, train_loader, device)
print(f"Training R²:  {train_r2:.6f}")
print(f"Training MSE: {train_mse:.6f}")
print()

# Compute R² for validation set
print("Computing R² for validation set...")
val_r2, val_mse, val_pred, val_target = evaluate_with_r2(model, val_loader, device)
print(f"Validation R²:  {val_r2:.6f}")
print(f"Validation MSE: {val_mse:.6f}")
print()

# Save results
results = {
    "model": log["model"],
    "timestamp": log["timestamp"],
    "data": {
        "total_samples": len(dataset),
        "train_samples": train_size,
        "val_samples": val_size
    },
    "metrics": {
        "train": {
            "r_squared": float(train_r2),
            "mse": float(train_mse)
        },
        "validation": {
            "r_squared": float(val_r2),
            "mse": float(val_mse)
        }
    },
    "comparison": {
        "train_vs_val_r2_diff": float(train_r2 - val_r2),
        "train_vs_val_mse_diff": float(train_mse - val_mse)
    }
}

# Save to JSON
output_path = os.path.join(RESULTS_DIR, "r_squared_results.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print("=" * 80)
print(f"Results saved to: {output_path}")
print()

# Print summary
print("SUMMARY")
print("=" * 80)
print(f"Model: {log['model']}")
print(f"Timestamp: {log['timestamp']}")
print()
print("R² Scores:")
print(f"  Training:   {train_r2:.6f}")
print(f"  Validation: {val_r2:.6f}")
print(f"  Difference: {train_r2 - val_r2:.6f}")
print()
print("MSE:")
print(f"  Training:   {train_mse:.6f}")
print(f"  Validation: {val_mse:.6f}")
print(f"  Difference: {train_mse - val_mse:.6f}")
print()

# Interpretation
print("INTERPRETATION:")
print("-" * 80)
if val_r2 > 0.95:
    print("✓ EXCELLENT: The model explains >95% of variance in the data")
elif val_r2 > 0.9:
    print("✓ VERY GOOD: The model explains >90% of variance in the data")
elif val_r2 > 0.8:
    print("✓ GOOD: The model explains >80% of variance in the data")
elif val_r2 > 0.7:
    print("○ MODERATE: The model explains >70% of variance in the data")
else:
    print("✗ POOR: The model explains <70% of variance in the data")
print()

if abs(train_r2 - val_r2) < 0.05:
    print("✓ Good generalization: Train and validation R² are very close")
elif abs(train_r2 - val_r2) < 0.1:
    print("○ Acceptable generalization: Small gap between train and val R²")
else:
    print("✗ Poor generalization: Large gap suggests overfitting")
print("=" * 80)
