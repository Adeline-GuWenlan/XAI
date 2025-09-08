# eval_moe_competitive.py
# Reload a checkpoint and print specialization/routing diagnostics.
# Mirrors the stats you were already looking at.

import json
import argparse
import torch
from torch.utils.data import DataLoader

from competitive_loss import (
    mixture_gaussian_nll, mixture_laplace_nll,
    entropy_from_probs, effective_num_experts
)

# --- Project-specific hooks (replace as needed) ---

def load_dataset(split: str, **kwargs):
    """
    Load quantum dataset for competitive MoE evaluation.
    Returns QuantumDataset that yields: (rho_real, rho_imag, magic_labels)
    """
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from Model_Archi.quantum_dataset import QuantumDataset
    
    # Default paths (can be overridden via kwargs)
    data_folder = kwargs.get('data_folder', 'Rawdata/DecodedTokens')
    labels_path = kwargs.get('labels_path', 'Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy')
    
    dataset = QuantumDataset(
        data_folder=data_folder,
        labels_path=labels_path,
        validate=False  # Skip validation for faster loading
    )
    
    return dataset

def build_model(**kwargs):
    """
    Build the CompleteQuantumMagicPredictor model from config.
    Returns model with MoE head that supports competitive evaluation.
    """
    import sys, os, json
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
    
    # Load default config (resolve path relative to XAI root)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../XAI/MLP_MoE/Competetive
    xai_root = os.path.dirname(os.path.dirname(script_dir))  # .../XAI (go up two levels)
    default_config = os.path.join(xai_root, 'CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json')
    config_path = kwargs.get('config_path', default_config)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override config with any provided kwargs
    config.update(kwargs)
    
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'],
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config['use_physics_mask'],
        mask_threshold=config['mask_threshold'],
        nhead=config['nhead'],
        use_cls_token=config['use_cls_token']
    )
    
    return model

def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f">> Loaded {ckpt_path}")

def forward_moe_debug(model, batch_data):
    """
    Forward pass through quantum magic model to extract MoE components.
    Handles both quantum data format (rho_real, rho_imag) and pre-processed tensors.
    
    Args:
        model: CompleteQuantumMagicPredictor with MoE head
        batch_data: Either (rho_real, rho_imag) tuple or pre-processed tensor
        
    Returns:
        yhat: (B, 1) final predictions
        gates: (B, E) gating weights
        expert_outs: (B, 1, E) expert outputs
    """
    # Handle different input formats
    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
        # Quantum data format: (rho_real, rho_imag)
        rho_real, rho_imag = batch_data
        
        # Forward through the full model to get embeddings
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
        
        # Apply pooling to get sequence representation
        if model.pooling_type == 'cls':
            pooled = tokens[:, 0, :]  # (B, d_model) - CLS token
        elif model.pooling_type == 'mean':
            pooled = tokens.mean(dim=1)  # (B, d_model)
        else:
            raise ValueError(f"Unsupported pooling type: {model.pooling_type}")
    else:
        # Assume pre-processed tensor
        pooled = batch_data
    
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

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = load_dataset("val", data_folder=args.data_folder, labels_path=args.labels_path)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()

    N, sum_mse, sum_nll = 0, 0.0, 0.0
    entropies = []
    usages = []
    per_expert_mse_top1_sum = None
    per_expert_counts = None

    for batch in loader:
        # Unpack quantum dataset format: (rho_real, rho_imag, labels)
        rho_real, rho_imag, yb = batch
        rho_real = rho_real.to(device)
        rho_imag = rho_imag.to(device)
        yb = yb.to(device)
        
        yhat, gates, expert_outs = forward_moe_debug(model, (rho_real, rho_imag))

        # metrics
        mse = ((yhat - yb) ** 2).mean().item()
        if args.lik == "gauss":
            nll = mixture_gaussian_nll(gates, expert_outs, yb, sigma=args.sigma).item()
        else:
            nll = mixture_laplace_nll(gates, expert_outs, yb, b=args.laplace_b).item()

        sum_mse += mse * rho_real.size(0)
        sum_nll += nll * rho_real.size(0)
        entropies.append(entropy_from_probs(gates).item())
        usages.append(gates.mean(dim=0, keepdim=True))  # (1,E)

        # Per-expert MSE @ top-1 responsibility
        top1 = gates.argmax(dim=-1)  # (B,)
        B, _, E = expert_outs.shape
        # gather the expert prediction picked by the gate
        idx = top1.view(B, 1, 1).expand(B, 1, 1)  # (B,1,1)
        picked = expert_outs.gather(2, idx).squeeze(-1)  # (B,1) from expert_outs (B,1,E)

        per_expert_err = (picked - yb) ** 2  # (B,1)
        # accumulate by expert id
        with torch.no_grad():
            onehot = torch.zeros(B, E, device=rho_real.device).scatter_(1, top1.unsqueeze(1), 1.0)  # (B,E)
            err_by_e = (per_expert_err * onehot.sum(dim=0).clamp_min(1e-9)).new_zeros(E)  # dummy to keep shape
            # Better: accumulate separately
            if per_expert_mse_top1_sum is None:
                per_expert_mse_top1_sum = torch.zeros(E, device=rho_real.device)
                per_expert_counts = torch.zeros(E, device=rho_real.device)
            # Add sums & counts
            per_expert_mse_top1_sum.index_add_(0, top1, per_expert_err.squeeze(1))
            per_expert_counts.index_add_(0, top1, torch.ones_like(top1, dtype=torch.float))

        N += rho_real.size(0)

    usage = torch.cat(usages, dim=0).mean(dim=0)  # (E,)
    per_expert_mse_top1 = (per_expert_mse_top1_sum / per_expert_counts.clamp_min(1)).tolist()

    report = {
        "N": N,
        "val_mse": sum_mse / max(N, 1),
        "val_nll": sum_nll / max(N, 1),
        "routing_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "usage_soft": usage.tolist(),
        "n_eff": float(effective_num_experts(usage.unsqueeze(0))),
        "per_expert_mse_top1": per_expert_mse_top1,
    }
    print(json.dumps(report, indent=2))

def parse_args():
    p = argparse.ArgumentParser("Evaluate competitive MoE checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_folder", type=str, default="Rawdata/DecodedTokens", help="Path to quantum data folder.")
    p.add_argument("--labels_path", type=str, default="Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy", help="Path to labels file.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lik", type=str, choices=["gauss","laplace"], default="gauss")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--laplace_b", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
