#!/usr/bin/env python3
"""
Run attention pooling analysis to extract attention-weighted vectors and MLP representations.
This mimics the hook behavior to capture vectors after "(matrix_tokens * attn_weights).sum(dim=1)"
and subsequent MLP layer outputs for correlation analysis with physical features.
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from Probing.Get_Layer_Represen.Atten_Prober.attention_pooling_probe import AttentionPoolingProbe


def load_model_and_data(model_path, data_folder, labels_path, device):
    """Load trained model and test data using QuantumDataset"""
    try:
        from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
        from Model_Archi.quantum_dataset import create_dataloaders
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default configuration - adjust based on your model
            config = {
                'matrix_dim': 4,
                'n_qubits': 2,
                'd_model': 128,
                'pooling_type': 'attention',  # Key: must be attention pooling
                'mlp_type': 'attention_enhanced',
                'use_physics_mask': False,
                'mask_threshold': 1,
                'nhead': 8,
                'use_cls_token': True
            }
        
        # Create model
        model = CompleteQuantumMagicPredictor(**config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        # Load test data using QuantumDataset
        _, _, test_loader = create_dataloaders(
            data_folder=data_folder,
            labels_path=labels_path,
            batch_size=32,
            train_ratio=0.8,
            val_ratio=0.1,
            num_workers=2,
            validate=False
        )
        
        return model, test_loader, config
        
    except Exception as e:
        print(f"Error loading model/data: {e}")
        print("Please check your model and data paths")
        return None, None, None


def analyze_attention_pooling(model_path, data_folder, labels_path, output_dir, max_samples=150000):
    """
    Extract attention pooling representations and save for correlation analysis
    
    Args:
        model_path: Path to trained model (.pth)
        data_folder: Path to folder containing _real.npy and _imag.npy files
        labels_path: Path to labels .npy file
        output_dir: Directory to save extracted features
        max_samples: Maximum number of samples to process
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    model, test_loader, config = load_model_and_data(model_path, data_folder, labels_path, device)
    if model is None:
        return
    
    print(f"Model configuration: {config}")
    
    # Check if model uses attention pooling
    if config.get('pooling_type') != 'attention':
        print(f"Warning: Model uses {config.get('pooling_type')} pooling, not attention pooling")
        print("This probe is specifically designed for attention pooling")
    
    # Create probe
    probe = AttentionPoolingProbe(model, device)
    
    # Extract representations from data loader
    all_representations = {}
    all_magic_values = []
    total_processed = 0
    
    print(f"Processing up to {max_samples} samples from data loader...")
    
    for batch_idx, (rho_real, rho_imag, magic_values) in enumerate(test_loader):
        if total_processed >= max_samples:
            break
            
        # Move to device
        rho_real = rho_real.to(device)
        rho_imag = rho_imag.to(device)
        
        # Limit batch size if needed
        remaining_samples = max_samples - total_processed
        current_batch_size = min(rho_real.shape[0], remaining_samples)
        
        if current_batch_size < rho_real.shape[0]:
            rho_real = rho_real[:current_batch_size]
            rho_imag = rho_imag[:current_batch_size]
            magic_values = magic_values[:current_batch_size]
        
        print(f"Processing batch {batch_idx + 1}, samples: {current_batch_size}")
        
        try:
            batch_representations = probe.extract_representations(rho_real, rho_imag)
            
            # Accumulate results
            for name, data in batch_representations.items():
                if name not in all_representations:
                    all_representations[name] = []
                all_representations[name].append(data)
            
            # Accumulate magic values
            all_magic_values.append(magic_values.cpu().numpy().flatten())
            total_processed += current_batch_size
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            continue
    
    # Concatenate all batches
    final_representations = {}
    for name, data_list in all_representations.items():
        if data_list:
            final_representations[name] = np.concatenate(data_list, axis=0)
    
    # Concatenate magic values
    if all_magic_values:
        magic_values_array = np.concatenate(all_magic_values)
    else:
        magic_values_array = None
    
    print(f"\nProcessed {total_processed} samples")
    print("Extracted representations:")
    for name, data in final_representations.items():
        print(f"  {name}: {data.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save representations to CSV
    probe.save_representations_csv(final_representations, output_dir, prefix="attention_pooling")
    
    # Save metadata
    metadata = {
        'model_config': config,
        'num_samples': total_processed,
        'representation_shapes': {name: data.shape for name, data in final_representations.items()},
        'device': str(device)
    }
    
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    # Save magic values if available
    if magic_values_array is not None:
        magic_df = pd.DataFrame({'magic_value': magic_values_array})
        magic_path = os.path.join(output_dir, "magic_values.csv")
        magic_df.to_csv(magic_path, index=False)
        print(f"Magic values saved to {magic_path}")
    
    print(f"\nAll results saved to {output_dir}")
    print("Ready for correlation analysis with physical features!")


def main():
    parser = argparse.ArgumentParser(description='Extract attention pooling representations')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--data_folder', type=str, 
                        default='Rawdata/DecodedTokens',
                        help='Path to folder containing _real.npy and _imag.npy files')
    parser.add_argument('--labels_path', type=str,
                        default='Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy',
                        help='Path to labels .npy file')
    parser.add_argument('--output_dir', type=str, default='./attention_pooling_results',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=150000,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    analyze_attention_pooling(args.model_path, args.data_folder, args.labels_path, args.output_dir, args.max_samples)


if __name__ == "__main__":
    # Example usage:
    # python run_attention_probe.py --model_path /path/to/model.pth --data_folder Rawdata/DecodedTokens --labels_path Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy
    main()