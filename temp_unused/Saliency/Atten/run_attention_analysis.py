#!/usr/bin/env python3
"""
Simplified script to run attention weight analysis on your trained quantum magic prediction model
"""
import os
import sys
import json

# Add Train directory to path
sys.path.append('Train')

# Set model and data paths based on your config
MODEL_PATH = "best_128_CLS_attention_enhanced_20250810_214428.pth"
CONFIG_PATH = "128_CLS_attention_enhanced_20250810_214428/128_CLS_attention_enhanced_config.json"

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Set data paths - update these to your actual data paths
DATA_FOLDER = config.get('data_folder', 'Rawdata/DecodedTokens')
LABELS_PATH = config.get('labels_path', 'Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy')

# Build command
cmd_parts = [
    "python run_saliency_analysis.py",
    f"--model_path {MODEL_PATH}",
    f"--data_folder {DATA_FOLDER}",
    f"--labels_path {LABELS_PATH}",
    f"--n_qubits {config['num_qubits']}",
    f"--d_model {config['d_model']}",
    f"--pooling_type {config['pooling_type']}",
    f"--mlp_type {config['mlp_type']}",
    f"--nhead {config['nhead']}",
    f"--mask_threshold {config['mask_threshold']}",
    "--max_samples 2048",
    "--individual_samples 20",
    "--include_attention",
    "--output_dir xai_analysis_results"
]

if config['use_physics_mask']:
    cmd_parts.append("--use_physics_mask")

cmd = " \\\n    ".join(cmd_parts)

print("Running XAI Analysis with the following command:")
print("=" * 80)
print(cmd)
print("=" * 80)

# Execute the command
os.system(" ".join(cmd_parts))