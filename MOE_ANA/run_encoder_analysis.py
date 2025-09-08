#!/usr/bin/env python3
"""
Run comprehensive encoder attention analysis on the quantum magic prediction model
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('/Users/guwenlan/Desktop/XAI')
sys.path.append('/Users/guwenlan/Desktop/XAI/Model_Archi')
sys.path.append('/Users/guwenlan/Desktop/XAI/MOE_ANA')

from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
from Model_Archi.quantum_dataset import QuantumDataset
from encoder_attention_analyzer import EncoderAttentionAnalyzer
from encoder_attention_visualizer import EncoderAttentionVisualizer

def load_model_and_config(model_path: str, config_path: str):
    """Load the trained model and its configuration"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'],
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config['use_physics_mask'],
        mask_threshold=config['mask_threshold'],
        nhead=config['nhead'],
        use_cls_token=config.get('use_cls_token', True)
    )
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {config['mlp_type']}")
    print(f"Number of encoder layers: {len(model.encoder_layers)}")
    print(f"Number of attention heads: {config['nhead']}")
    print(f"Device: {device}")
    
    return model, config, device

def create_dataloader(config, batch_size=32, max_samples=500):
    """Create dataloader for analysis using correct 50K labels"""
    # Dataset parameters
    data_folder = Path("/Users/guwenlan/Desktop/XAI") / config['data_folder']
    # Use the correct matching 50K label file
    labels_path = Path("/Users/guwenlan/Desktop/XAI/Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_50000_datapoints.npy")
    
    print(f"Loading data from: {data_folder}")
    print(f"Loading labels from: {labels_path}")
    
    # Create dataset with matching 50K labels
    dataset = QuantumDataset(
        data_folder=str(data_folder),
        labels_path=str(labels_path),
        validate=True  # Enable validation now that we have matching data/labels
    )
    
    # Get actual available data size
    actual_size = min(len(dataset), max_samples)
    print(f"Using {actual_size} samples from dataset")
    
    # Create random subset for analysis
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), actual_size, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print(f"Final dataset size: {len(dataset)} samples")
    
    return dataloader

def main():
    # Configuration
    model_path = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth"
    config_path = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json"
    output_dir = "/Users/guwenlan/Desktop/XAI/MOE_ANA/encoder_results"
    
    print("=" * 80)
    print("ENCODER ATTENTION ANALYSIS - QUANTUM MAGIC PREDICTION MODEL")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and config
    print("\n1. Loading model and configuration...")
    model, config, device = load_model_and_config(model_path, config_path)
    
    # Create dataloader
    print("\n2. Creating dataloader...")
    dataloader = create_dataloader(config, batch_size=16, max_samples=500)
    
    # Initialize analyzer
    print("\n3. Initializing encoder attention analyzer...")
    analyzer = EncoderAttentionAnalyzer(model, device, config)
    
    # Initialize visualizer
    visualizer = EncoderAttentionVisualizer(output_dir)
    
    # Generate quantum labels
    n_qubits = config['num_qubits']
    quantum_labels = visualizer.generate_quantum_labels(n_qubits)
    
    # Run encoder attention analysis
    print("\n4. Running comprehensive encoder attention analysis...")
    print("This will analyze:")
    print("- CLS token attention patterns across encoder layers")
    print("- Head specialization within each layer")
    print("- Attention evolution through encoder layers") 
    print("- Cross-layer head comparison")
    print("- Individual sample attention patterns")
    
    analysis_results = analyzer.analyze_dataset_encoder_attention(dataloader, max_samples=500)
    
    print(f"\nAnalysis completed for {len(analysis_results['sample_data'])} samples")
    
    # Create visualizations
    print("\n5. Creating encoder attention visualizations...")
    visualizer.create_encoder_attention_dashboard(analysis_results, quantum_labels)
    
    # Print summary
    print("\n6. Encoder Attention Analysis Summary:")
    print("=" * 60)
    
    summary = analysis_results.get('summary_stats', {})
    print(f"Mean Absolute Error: {summary['avg_prediction_error']:.4f}")
    print(f"Prediction Range: [{summary['predicted_magic_range'][0]:.4f}, {summary['predicted_magic_range'][1]:.4f}]")
    print(f"True Range: [{summary['true_magic_range'][0]:.4f}, {summary['true_magic_range'][1]:.4f}]")
    
    # Layer evolution summary
    layer_evolution = analysis_results.get('layer_evolution', [])
    if layer_evolution:
        print("\nEncoder Layer Evolution:")
        for layer_stat in layer_evolution:
            print(f"  Layer {layer_stat['layer']}: "
                  f"Entropy={layer_stat['mean_entropy']:.3f}±{layer_stat['std_entropy']:.3f}, "
                  f"Sparsity={layer_stat['mean_sparsity']:.3f}±{layer_stat['std_sparsity']:.3f}")
    
    # Head specialization summary
    head_analysis = analysis_results.get('head_specialization', {})
    if head_analysis:
        print(f"\nHead Specialization Analysis:")
        for layer_idx, layer_heads in head_analysis.items():
            consistencies = [head['consistency'] for head in layer_heads.values()]
            entropies = [head['mean_entropy'] for head in layer_heads.values()]
            print(f"  Layer {layer_idx}: "
                  f"Avg Head Consistency={np.mean(consistencies):.3f}, "
                  f"Avg Head Entropy={np.mean(entropies):.3f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles generated:")
    print("- encoder_attention_analysis_report.txt: Comprehensive analysis report")
    print("- encoder_layer_evolution_statistics.png: Evolution through encoder layers")
    print("- head_specialization_layer_*.png: Head specialization per layer")  
    print("- head_*_across_layers.png: Same head across different layers")
    print("- cls_attention_evolution_sample_*.png: Individual CLS attention patterns")
    print("- attention_rollout_sample_*.png: Information flow analysis")
    
    print("\n" + "=" * 80)
    print("ENCODER ATTENTION ANALYSIS COMPLETE!")
    print("Focus: Transformer encoder layers, attention heads, CLS patterns")
    print("=" * 80)

if __name__ == "__main__":
    main()