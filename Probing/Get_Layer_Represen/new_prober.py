# probe_extract_csv_exact.py
import os
import numpy as np
import pandas as pd
import torch
import sys
import time
from torch.utils.data import DataLoader

from pathlib import Path
import sys

# repo_root is the parent directory that contains 'Model_Archi' and 'Probing'
repo_root = Path(__file__).resolve().parents[2]  # adjust if your file depth differs
sys.path.insert(0, str(repo_root))

from Model_Archi.quantum_dataset import QuantumDataset
from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
from Probing.Get_Layer_Represen.hooks_regist_only import LayerwiseCLSProbeExact


def _normalize_trace(r_real: torch.Tensor, r_imag: torch.Tensor, eps: float = 1e-12):
    """
    Per-sample trace normalization: ρ <- ρ / Tr(ρ).
    Assumes r_real is the real part; trace = sum(diagonal of r_real).
    """
    diag_real = torch.diagonal(r_real, dim1=1, dim2=2)  # [B, D]
    tr = diag_real.sum(dim=1)                           # [B]
    tr = torch.clamp(tr, min=eps)
    scale = (1.0 / tr).view(-1, 1, 1)
    r_real.mul_(scale)
    r_imag.mul_(scale)
    return r_real, r_imag

def build_loader_sequential(data_folder, labels_path, batch_size=512, num_workers=4, validate=True, device_type='cuda'):
    ds = QuantumDataset(data_folder=data_folder, labels_path=labels_path, validate=validate)
    
    # Optimize DataLoader based on claude.md recommendations
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=min(num_workers, os.cpu_count()) if num_workers > 0 else 0,
        pin_memory=(device_type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    return ds, loader

def extract_layer_csv_exact(model_cfg: dict, ckpt_path: str,
                            data_folder: str, labels_path: str,
                            out_dir: str, device: str = "cuda:0",
                            batch_size: int = 1024, num_workers: int = 8,
                            normalize_trace: bool = True):
    """
    Export one CSV per fixed layer key defined by LayerwiseCLSProbeExact.KEY_ORDER.
    Each CSV has columns: idx, f0..f{d-1}
    GPU-optimized version following claude.md recommendations.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Device setup with proper CUDA initialization
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f'Device: {dev}, GPUs available: {torch.cuda.device_count()}')
    
    # Enable cudnn benchmark for better GPU performance
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1) Model (exact config, no guesses)
    model = CompleteQuantumMagicPredictor(**model_cfg)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
        print(f"✅ Loaded checkpoint: {ckpt_path}")
    else:
        print("⚠️ Checkpoint not found. Using randomly initialized weights.")
    
    # Move model to device and verify placement
    model.to(dev).eval()
    print(f'Model on device: {next(model.parameters()).device}')

    # 2) Optimized DataLoader
    ds, loader = build_loader_sequential(data_folder, labels_path, batch_size, num_workers, 
                                         validate=True, device_type=dev.type)

    # 3) Prober with dynamic keys - this handles model device placement internally
    prober = LayerwiseCLSProbeExact(model, dev)
    # Get the expected keys for this specific model architecture
    keys = prober.get_expected_keys()
    print(f"Expected keys for architecture: {keys[:5]}... ({len(keys)} total)")

    # banks
    bank = {k: [] for k in keys}
    idx_bank = []

    # 4) GPU-optimized iteration with proper inference mode
    print(f"Processing {len(ds)} samples in batches of {batch_size}")
    start_idx = 0
    batch_times = []
    
    # Use inference_mode for better GPU performance
    with torch.inference_mode():
        for batch_i, (rho_real, rho_imag, _) in enumerate(loader):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            batch_start = time.time()
            
            B = rho_real.size(0)
            if normalize_trace:
                rho_real, rho_imag = _normalize_trace(rho_real, rho_imag)

            # Move to device with non_blocking for better throughput
            rho_real = rho_real.to(dev, non_blocking=True)
            rho_imag = rho_imag.to(dev, non_blocking=True)

            cache = prober.run_once(rho_real, rho_imag)  # dynamic dict based on architecture

            # Handle dynamic keys - cache might have different keys than expected
            for k in cache.keys():
                if k not in bank:
                    bank[k] = []  # Add new key if found
                bank[k].append(cache[k])
                
            idx_bank.append(np.arange(start_idx, start_idx + B, dtype=np.int64))
            start_idx += B
            
            # Time tracking for performance monitoring
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_i % 10 == 0:
                avg_time = np.mean(batch_times[-10:])
                print(f"Batch {batch_i}/{len(loader)}: {avg_time:.3f}s/batch")

    # Performance summary
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        total_samples = start_idx
        throughput = total_samples / sum(batch_times)
        print(f"\n📊 Performance Summary:")
        print(f"   Average batch time: {avg_batch_time:.3f}s")
        print(f"   Throughput: {throughput:.1f} samples/sec")
        print(f"   Total samples processed: {total_samples}")

    idx_all = np.concatenate(idx_bank, axis=0)

    # 5) Write one CSV per layer (using actual keys from bank)
    manifest_rows = []
    actual_keys = list(bank.keys())
    print(f"Writing CSVs for {len(actual_keys)} layers: {actual_keys}")
    
    for k in actual_keys:
        if bank[k]:  # Only process if we have data
            feat = np.concatenate(bank[k], axis=0)  # [N, d]
            cols = [f"f{i}" for i in range(feat.shape[1])]
            df = pd.DataFrame(feat, columns=cols)
            df.insert(0, "idx", idx_all)
            out_csv = os.path.join(out_dir, f"{k}.csv")
            df.to_csv(out_csv, index=False)
            manifest_rows.append({"layer": k, "path": out_csv, "dim": feat.shape[1]})
            print(f"💾 {k}: {feat.shape} -> {out_csv}")

    pd.DataFrame(manifest_rows).to_csv(os.path.join(out_dir, "manifest_layers.csv"), index=False)
    print(f"✅ Done. {len(actual_keys)} CSVs written under '{out_dir}'")
