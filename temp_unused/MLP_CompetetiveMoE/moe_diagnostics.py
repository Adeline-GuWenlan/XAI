#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moe_diagnostics.py

Purpose
-------
Load a frozen MoE-style model checkpoint and a small batch of input/labels,
then *inspect* current behavior under your existing MSE objective—without
nudging the gate (no regularizers, no temperature tricks).

It reports:
  • Utilization (top-1 usage per expert, entropy, effective #experts)
  • Routing entropy (per-sample, with mean/std/quantiles)
  • Contribution per expert (avg |π_j * f_j|)
  • Per-expert conditional error (MSE if routed to j via top-1)
  • Overall MSE of the *mixture* prediction

Assumptions
-----------
Your model exposes the MoE head as:
    model.mlp_head['gating_network']  -> nn.Module (softmax output shape [B, E])
    model.mlp_head['experts']         -> nn.ModuleList of E experts
and uses a regression head producing shape [B, 1] per expert.

If your checkpoint is a full torch-saved Module, we’ll load it directly.
If it’s a state_dict, you must specify --model-import and --model-class so we can instantiate the model skeleton before loading weights.

Data formats supported:
  • .npz or .npy  with keys: X (N, d_model), y (N,) or (N,1)
  • .pt           dict with 'X' and 'y' tensors
  • .csv          needs --label-col (feature columns = all others); numeric only

Usage
-----
python moe_diagnostics.py \
  --checkpoint /path/to/ckpt.pt \
  --data /path/to/data.npz \
  --limit 2048 \
  --device auto

If checkpoint is a state_dict and you need to build the model class:
python moe_diagnostics.py \
  --checkpoint /path/to/state_dict.pt \
  --model-import mypkg.models.my_model \
  --model-class MagicNet \
  --model-kwargs '{"d_model":128,"dropout_rate":0.1}' \
  --data /path/to/data.csv \
  --label-col target

Notes
-----
• No gradients, model.eval(), torch.no_grad().
• No architectural changes; we just *call* your gating + experts to inspect.
"""

from __future__ import annotations
import argparse, importlib, json, math, os, sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add path for model and dataset imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model_Archi.quantum_dataset import QuantumDataset
from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor


# ---------------------------
# Utils
# ---------------------------

def set_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)

def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def load_quantum_data(data_folder: str, labels_path: str, limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load quantum data using the QuantumDataset class.
    Returns processed embeddings and labels for MoE analysis.
    """
    dataset = QuantumDataset(
        data_folder=data_folder,
        labels_path=labels_path,
        validate=False  # Skip validation for speed
    )
    
    # Limit the number of samples
    n_samples = min(limit, len(dataset))
    print(f"Loading {n_samples} samples for MoE analysis...")
    
    # Create dummy model to process samples into embeddings
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
    temp_config = {
        'matrix_dim': 4,
        'num_qubits': 2,
        'd_model': 128,
        'pooling_type': 'cls',
        'mlp_type': 'mixture_of_experts',
        'use_physics_mask': False,
        'mask_threshold': 1,
        'nhead': 8
    }
    
    # We'll extract embeddings from the full forward pass
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        rho_real, rho_imag, label = dataset[i]
        # Store raw matrices - we'll process through the encoder later
        X_list.append((rho_real, rho_imag))
        y_list.append(label.item())
    
    return X_list, np.array(y_list)

def load_pt(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "X" in obj and "y" in obj:
        return obj["X"], obj["y"]
    raise ValueError("Expected a .pt dict with keys 'X' and 'y'.")

def load_csv(path: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"--label-col '{label_col}' not found in CSV.")
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()
    return X, y

def standardize_y_shape(y: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return y.float()

def mse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(yhat, y).detach().cpu().item())

def safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return (x + eps).log()

# ---------------------------
# Model loading
# ---------------------------

def build_quantum_magic_model(config: Dict) -> nn.Module:
    """
    Build the CompleteQuantumMagicPredictor model from config.
    """
    return CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'],
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config['use_physics_mask'],
        mask_threshold=config['mask_threshold'],
        nhead=config['nhead']
    )

def load_quantum_model(checkpoint: str, config: Dict, device: torch.device) -> nn.Module:
    """
    Load the quantum magic prediction model with MoE head.
    """
    # Build model from config
    model = build_quantum_magic_model(config)
    
    # Load state dict
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    for p in model.parameters(): 
        p.requires_grad_(False)
    
    return model

# ---------------------------
# MoE probing (no training)
# ---------------------------

@torch.no_grad()
def forward_moe_debug(model: nn.Module, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass through the model to extract MoE behavior.
    
    Args:
        model: CompleteQuantumMagicPredictor with MoE head
        rho_real: (B, D, D) real parts of density matrices
        rho_imag: (B, D, D) imaginary parts of density matrices
    
    Returns:
        yhat: (B, 1) final predictions
        gates: (B, E) gating weights
        outs: (B, 1, E) expert outputs
    """
    # First get embeddings from the encoder
    embeddings = model.embedding(rho_real, rho_imag)  # (B, seq_len, d_model)
    
    # Add CLS token if using CLS pooling
    B = embeddings.shape[0]
    if model.pooling_type == "cls" and hasattr(model, 'cls_token'):
        cls_tokens = model.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls_tokens, embeddings], dim=1)  # [B, D²+1, d_model]
    else:
        tokens = embeddings
    
    # Process through transformer layers
    for layer in model.encoder_layers:
        tokens = layer(tokens)
    
    encoded = tokens
    
    # Apply pooling to get sequence representation
    if model.pooling_type == 'cls':
        pooled = encoded[:, 0, :]  # (B, d_model) - CLS token
    elif model.pooling_type == 'mean':
        pooled = encoded.mean(dim=1)  # (B, d_model)
    else:
        raise ValueError(f"Unsupported pooling type: {model.pooling_type}")
    
    # Check if model has MoE structure
    if not hasattr(model, "mlp_head") or model.mlp_head.mlp_type != "mixture_of_experts":
        raise AttributeError(f"Model should have MoE mlp_head, got {model.mlp_head.mlp_type}")
    
    # Access MoE components from the mlp_head module
    moe_components = model.mlp_head.mlp_head  # The ModuleDict inside QuantumMagicMLPv2
    gating = moe_components["gating_network"]
    experts = moe_components["experts"]
    
    # Get gating weights and expert outputs
    gates = gating(pooled)  # (B, E) - already softmax
    expert_outputs = torch.stack([expert(pooled) for expert in experts], dim=-1)  # (B, 1, E)
    
    # Compute weighted mixture
    yhat = (expert_outputs * gates.unsqueeze(1)).sum(dim=-1)  # (B, 1)
    
    return yhat, gates, expert_outputs

def routing_entropy(gates: torch.Tensor) -> torch.Tensor:
    """
    Mean routing entropy over batch:
      H_route(x) = -sum_j π_j(x) log π_j(x)
    """
    H = -(gates * safe_log(gates)).sum(dim=-1)  # (B,)
    return H.mean()

def utilization_stats(gates: torch.Tensor) -> Dict[str, float | np.ndarray]:
    """
    Utilization measured by top-1 routing and by soft usage.
    Returns dict with entropy, normalized entropy, effective experts, CV, and shares.
    """
    B, E = gates.shape
    # Top-1 hard assignment
    assigns = gates.argmax(dim=-1)                         # (B,)
    counts = torch.bincount(assigns, minlength=E).float()  # (E,)
    u_top1 = counts / counts.sum()

    # Soft usage (expected counts)
    u_soft = gates.mean(dim=0)                             # (E,)

    def ent(u: torch.Tensor) -> float:
        Hu = float((-(u * safe_log(u)).sum()).cpu().item())
        return Hu

    H_top1 = ent(u_top1)
    H_soft = ent(u_soft)
    logE   = math.log(E)

    stats = {
        "E": E,
        "u_top1": u_top1.cpu().numpy(),
        "u_soft": u_soft.cpu().numpy(),
        "H_util_top1": H_top1,
        "H_util_top1_norm": H_top1 / logE if E > 1 else 1.0,
        "N_eff_top1": math.exp(H_top1),
        "H_util_soft": H_soft,
        "H_util_soft_norm": H_soft / logE if E > 1 else 1.0,
        "N_eff_soft": math.exp(H_soft),
        "CV_top1": float(u_top1.std(unbiased=False) / (u_top1.mean() + 1e-12)),
        "CV_soft": float(u_soft.std(unbiased=False) / (u_soft.mean() + 1e-12)),
    }
    return stats

def contribution_per_expert(gates: torch.Tensor, outs: torch.Tensor) -> np.ndarray:
    """
    Average absolute contribution |π_j * f_j| across batch.
    outs: (B, 1, E) ; gates: (B, E)
    Returns (E,)
    """
    contrib = (outs.squeeze(1) * gates).abs().mean(dim=0)  # (E,)
    return contrib.detach().cpu().numpy()

def per_expert_mse_top1(outs: torch.Tensor, y: torch.Tensor, gates: torch.Tensor) -> np.ndarray:
    """
    For samples where expert j is top-1, compute MSE(f_j(x), y).
    Returns (E,) with NaN for experts that received no samples.
    """
    B, _, E = outs.shape
    assigns = gates.argmax(dim=-1)  # (B,)
    mse_list = []
    for j in range(E):
        mask = (assigns == j).unsqueeze(1)  # (B,1)
        if mask.any():
            fj = outs[:, :, j]              # (B,1)
            diff = fj[mask] - y[mask]
            val = float((diff.pow(2).mean()).cpu().item())
        else:
            val = float("nan")
        mse_list.append(val)
    return np.array(mse_list, dtype=np.float64)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="MoE diagnostics (frozen model, no training).")
    ap.add_argument("--checkpoint", type=str, default="/Users/guwenlan/Desktop/XAI/CONFIGs/moe_competitive_00000019.pt", help="Path to model checkpoint (.pt)")
    ap.add_argument("--config", type=str, default="CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json", help="Path to model config JSON")
    ap.add_argument("--data-folder", type=str, default="Rawdata/DecodedTokens", help="Path to quantum data folder")
    ap.add_argument("--labels-path", type=str, default="Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy", help="Path to labels")
    ap.add_argument("--limit", type=int, default=4096, help="Max #examples to load for quick diagnostics.")
    ap.add_argument("--batch-size", type=int, default=1024, help="Batch size for forward pass.")
    ap.add_argument("--device", type=str, default="auto", help="cpu | cuda | mps | auto")
    ap.add_argument("--save-json", type=str, default=None, help="Optional path to save metrics as JSON.")
    args = ap.parse_args()

    device = set_device(args.device)

    # Load model config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Loading model with config: {config['mlp_type']}")
    
    # Load model
    model = load_quantum_model(args.checkpoint, config, device)
    
    # Load quantum data
    X_list, y_np = load_quantum_data(args.data_folder, args.labels_path, args.limit)
    
    # Prepare data
    N = len(X_list)
    y_t = standardize_y_shape(y_np).to(device)
    
    print(f"Loaded {N} quantum samples for MoE analysis")

    # Forward in batches
    all_gates = []
    all_outs  = []
    all_yhat  = []
    all_y     = []

    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print(f"Processing {N} samples in batches of {args.batch_size}...")
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(N, start + args.batch_size)
            
            # Prepare batch of quantum matrices
            batch_real = []
            batch_imag = []
            for i in range(start, end):
                rho_real, rho_imag = X_list[i]
                batch_real.append(rho_real)
                batch_imag.append(rho_imag)
            
            batch_real = torch.stack(batch_real).to(device)
            batch_imag = torch.stack(batch_imag).to(device)
            yb = y_t[start:end]
            
            # Forward through MoE
            yhat, gates, outs = forward_moe_debug(model, batch_real, batch_imag)
            all_gates.append(gates)
            all_outs.append(outs)
            all_yhat.append(yhat)
            all_y.append(yb)
            
            if (start // args.batch_size + 1) % 10 == 0:
                print(f"  Processed batch {start // args.batch_size + 1}/{(N + args.batch_size - 1) // args.batch_size}")

    gates = torch.cat(all_gates, dim=0)    # (N, E)
    outs  = torch.cat(all_outs, dim=0)     # (N, 1, E)
    yhat  = torch.cat(all_yhat, dim=0)     # (N, 1)
    y     = torch.cat(all_y, dim=0)        # (N, 1)

    # Metrics
    E = gates.shape[1]
    overall_mse = mse(yhat, y)
    route_H     = float(routing_entropy(gates).cpu().item())

    # Routing entropy distribution
    per_sample_H = (-(gates * safe_log(gates)).sum(dim=-1)).detach().cpu().numpy()
    H_mean = float(per_sample_H.mean())
    H_std  = float(per_sample_H.std())
    H_q    = {q: float(np.quantile(per_sample_H, q/100.0)) for q in [0, 10, 25, 50, 75, 90, 100]}

    util     = utilization_stats(gates)
    contrib  = contribution_per_expert(gates, outs)              # (E,)
    mse_top1 = per_expert_mse_top1(outs, y, gates)               # (E,)

    # Pretty print
    logE = math.log(E) if E > 1 else 1.0
    print("\n================= MoE Diagnostics (Frozen) =================")
    print(f"Device: {device.type} | N={N} | Experts E={E}")
    print(f"Overall mixture MSE: {overall_mse:.6f}")
    print("\n-- Routing Entropy --")
    print(f"Mean H_route: {H_mean:.4f}  (normalized: {H_mean/logE:.4f})")
    print(f"Std: {H_std:.4f}")
    print("Quantiles:", {k: f"{v:.4f}" for k, v in H_q.items()})

    print("\n-- Utilization (Top-1 and Soft) --")
    print(f"H_util_top1: {util['H_util_top1']:.4f}  (norm: {util['H_util_top1_norm']:.4f})  "
          f"N_eff_top1: {util['N_eff_top1']:.2f}  CV_top1: {util['CV_top1']:.3f}")
    print(f"H_util_soft: {util['H_util_soft']:.4f}  (norm: {util['H_util_soft_norm']:.4f})  "
          f"N_eff_soft: {util['N_eff_soft']:.2f}  CV_soft: {util['CV_soft']:.3f}")

    def row(arr, fmt="{:.4f}"):
        return "[" + ", ".join(fmt.format(x) for x in arr) + "]"

    print(f"u_top1 (share): {row(util['u_top1'])}")
    print(f"u_soft (mean π): {row(util['u_soft'])}")

    print("\n-- Contribution per Expert (avg |π_j * f_j|) --")
    print(row(contrib))

    print("\n-- Per-Expert MSE on Top-1 Assigned Samples --")
    print(row(mse_top1, fmt="{:.6f}"))
    print("============================================================\n")

    if args.save_json:
        out = {
            "N": int(N),
            "E": int(E),
            "overall_mse": overall_mse,
            "routing_entropy_mean": H_mean,
            "routing_entropy_std": H_std,
            "routing_entropy_quantiles": H_q,
            "utilization": {
                "u_top1": util["u_top1"].tolist(),
                "u_soft": util["u_soft"].tolist(),
                "H_util_top1": util["H_util_top1"],
                "H_util_top1_norm": util["H_util_top1_norm"],
                "N_eff_top1": util["N_eff_top1"],
                "CV_top1": util["CV_top1"],
                "H_util_soft": util["H_util_soft"],
                "H_util_soft_norm": util["H_util_soft_norm"],
                "N_eff_soft": util["N_eff_soft"],
                "CV_soft": util["CV_soft"],
            },
            "contribution_abs": contrib.tolist(),
            "per_expert_mse_top1": mse_top1.tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics JSON → {args.save_json}")


if __name__ == "__main__":
    main()
