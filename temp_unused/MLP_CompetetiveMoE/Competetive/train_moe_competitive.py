# train_moe_competitive.py
# Fine-tune your *existing* MoE checkpoint with competitive mixture NLL.
# ---------------------------------------------------------
# Minimal wiring; plug in your dataset/model paths in the TODOs.

import os, math, json, time, argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from competitive_loss import (
    mixture_gaussian_nll, mixture_laplace_nll,
    balance_kl, entropy_from_probs, effective_num_experts
)

# ------------------------
# Utilities / placeholders
# ------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark=True

def load_dataset(split: str, **kwargs):
    """
    Load quantum dataset for competitive MoE training.
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
    
    # For now, return the full dataset (splitting will be handled by DataLoader)
    # In production, implement proper train/val/test splits
    return dataset

def build_model(**kwargs):
    """
    Build the CompleteQuantumMagicPredictor model from config.
    Returns model with MoE head that supports competitive training.
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
    """
    Loads your existing collaborative-MoE checkpoint.
    """
    if ckpt_path is None or ckpt_path == "":
        print(">> No checkpoint path provided; training from current weights.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Adapt the key according to your saver format
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f">> Loaded checkpoint: {ckpt_path}")
    if missing:   print("   missing keys:", missing)
    if unexpected:print("   unexpected:", unexpected)

def save_checkpoint(model, out_dir, step_or_epoch, extra=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"moe_competitive_{step_or_epoch:08d}.pt"
    payload = {"model": model.state_dict()}
    if extra: payload["extra"] = extra
    torch.save(payload, save_path)
    return str(save_path)

# ---------------------------------------------------------
# Forward adapter
# ---------------------------------------------------------

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
    
    # Validate shapes
    assert gates.dim() == 2, f"gates must be (B,E), got {gates.shape}"
    assert expert_outputs.dim() == 3 and expert_outputs.shape[1] == 1, f"expert_outs must be (B,1,E), got {expert_outputs.shape}"
    assert yhat.dim() == 2 and yhat.shape[1] == 1, f"yhat must be (B,1), got {yhat.shape}"
    
    return yhat, gates, expert_outputs

# ---------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------

@torch.no_grad()
def eval_epoch(model, loader, device, lik_type, sigma, b, lb_lambda):
    model.eval()
    n, total_mse, total_nll = 0, 0.0, 0.0
    entropies, all_usage = [], []
    for batch in loader:
        # Unpack quantum dataset format: (rho_real, rho_imag, labels)
        rho_real, rho_imag, yb = batch
        rho_real = rho_real.to(device)
        rho_imag = rho_imag.to(device)
        yb = yb.to(device)
        
        yhat, gates, expert_outs = forward_moe_debug(model, (rho_real, rho_imag))
        mse = ((yhat - yb) ** 2).mean().item()

        if lik_type == "gauss":
            nll = mixture_gaussian_nll(gates, expert_outs, yb, sigma=sigma).item()
        else:
            nll = mixture_laplace_nll(gates, expert_outs, yb, b=b).item()

        total_mse += mse * rho_real.size(0)
        total_nll += nll * rho_real.size(0)
        entropies.append(entropy_from_probs(gates).item())
        all_usage.append(gates.mean(dim=0, keepdim=True))  # (1,E)
        n += rho_real.size(0)

    usage = torch.cat(all_usage, dim=0).mean(dim=0)  # (E,)
    stats = {
        "mse": total_mse / max(n, 1),
        "nll": total_nll / max(n, 1),
        "route_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "usage": usage.tolist(),
        "n_eff": float(effective_num_experts(usage.unsqueeze(0)))
    }
    return stats

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # -- Data
    train_ds = load_dataset("train", data_folder=args.data_folder, labels_path=args.labels_path)
    val_ds   = load_dataset("val", data_folder=args.data_folder, labels_path=args.labels_path)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    # -- Model
    model = build_model()
    load_checkpoint(model, args.init_ckpt)
    model.to(device)

    # -- Optimizer / AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp))

    # -- Schedules for competition: tau (temperature) & top-k
    def anneal_tau(epoch):
        # Linear anneal from tau_start -> tau_end over tau_warmup epochs
        if args.tau_warmup <= 0:
            return args.tau_end
        t = min(epoch / max(1, args.tau_warmup), 1.0)
        return (1 - t) * args.tau_start + t * args.tau_end

    def anneal_topk(epoch):
        # Switch to topk_target at epoch >= topk_start
        return None if epoch < args.topk_start else args.topk_target

    best_val, best_path = math.inf, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = anneal_tau(epoch)
        tk  = anneal_topk(epoch)

        running = {"mse": 0.0, "nll": 0.0, "lb": 0.0}
        nseen = 0

        for batch in train_loader:
            # Unpack quantum dataset format: (rho_real, rho_imag, labels)
            rho_real, rho_imag, yb = batch
            rho_real = rho_real.to(device)
            rho_imag = rho_imag.to(device) 
            yb = yb.to(device)

            with torch.cuda.amp.autocast(enabled=(not args.no_amp)):
                yhat, gates, expert_outs = forward_moe_debug(model, (rho_real, rho_imag))

                # Competitive NLL
                if args.lik == "gauss":
                    loss_main = mixture_gaussian_nll(
                        gates, expert_outs, yb, sigma=args.sigma, tau=tau, topk=tk
                    )
                else:
                    loss_main = mixture_laplace_nll(
                        gates, expert_outs, yb, b=args.laplace_b, tau=tau, topk=tk
                    )

                # Optional load-balance regularizer
                loss_lb = args.lb_lambda * balance_kl(gates) if args.lb_lambda > 0 else 0.0
                loss = loss_main + loss_lb

            scaler.scale(loss).backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            # Logging (on-the-fly)
            mse_batch = ((yhat - yb) ** 2).mean().item()
            running["mse"] += mse_batch * rho_real.size(0)
            running["nll"] += float(loss_main.detach().cpu()) * rho_real.size(0)
            running["lb"]  += (float(loss_lb) if isinstance(loss_lb, float) else float(loss_lb.detach().cpu())) * rho_real.size(0)
            nseen += rho_real.size(0)

        train_logs = {k: v / max(nseen, 1) for k, v in running.items()}
        val_logs = eval_epoch(model, val_loader, device,
                              lik_type=args.lik, sigma=args.sigma, b=args.laplace_b,
                              lb_lambda=args.lb_lambda)

        wall = time.time() - start
        
        # Handle val_logs with potential list values (like usage)
        val_formatted = {}
        for k, v in val_logs.items():
            if isinstance(v, list):
                val_formatted[f"val_{k}"] = [round(x, 6) if isinstance(x, (int, float)) else x for x in v]
            else:
                val_formatted[f"val_{k}"] = round(v, 6)
        
        logline = {
            "epoch": epoch, "wall_s": round(wall, 1),
            "tau": round(tau, 4), "topk": tk,
            **{f"train_{k}": round(v, 6) for k, v in train_logs.items()},
            **val_formatted
        }
        print(json.dumps(logline))

        # Save best by validation NLL (or MSE if you prefer)
        metric = val_logs["nll"]
        if metric < best_val:
            best_val = metric
            best_path = save_checkpoint(model, args.out_dir, step_or_epoch=epoch,
                                        extra={"val_nll": best_val, "epoch": epoch})
            print(f">> Saved new best to {best_path}")

    print(">> Done. Best val NLL:", best_val, "path:", best_path)

def parse_args():
    p = argparse.ArgumentParser("Competitive MoE Fine-tune")
    # Paths / I/O
    p.add_argument("--init_ckpt", type=str, default="", help="Path to existing (collab) MoE checkpoint.")
    p.add_argument("--out_dir", type=str, default="./ckpts_comp", help="Where to save fine-tuned checkpoints.")
    p.add_argument("--data_folder", type=str, default="Rawdata/DecodedTokens", help="Path to quantum data folder.")
    p.add_argument("--labels_path", type=str, default="Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy", help="Path to labels file.")
    p.add_argument("--cpu", action="store_true")
    # Data
    p.add_argument("--bs", type=int, default=128)
    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--no_amp", action="store_true")
    # Likelihood & regularization
    p.add_argument("--lik", type=str, choices=["gauss","laplace"], default="gauss")
    p.add_argument("--sigma", type=float, default=0.1, help="Gaussian sigma")
    p.add_argument("--laplace_b", type=float, default=0.1, help="Laplace scale b")
    p.add_argument("--lb_lambda", type=float, default=0.01, help="Load-balance KL weight")
    # Competition controls (applied *in loss only*)
    p.add_argument("--tau_start", type=float, default=1.8, help="Initial temperature (>1 softer)")
    p.add_argument("--tau_end",   type=float, default=1.0, help="Final temperature (≈1 sharp)")
    p.add_argument("--tau_warmup", type=int, default=10, help="Epochs to anneal tau")
    p.add_argument("--topk_start", type=int, default=15, help="Enable top-k from this epoch (1-indexed)")
    p.add_argument("--topk_target", type=int, default=1, help="k for top-k gating in loss")
    # Misc
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
