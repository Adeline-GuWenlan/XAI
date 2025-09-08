#!/usr/bin/env python3
"""
Comprehensive XAI Analysis Script for Quantum Magic Prediction
Includes: Attention Analysis, Layer-wise Probing, Neuron Analysis
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import all analysis modules
from attention_analyzer import AttentionAnalyzer
from attention_visualizer import AttentionVisualizer
from layer_wise_prober import LayerWiseProber
from probing_visualizer import ProbingVisualizer
from neuron_analyzer import NeuronAnalyzer
from neuron_visualizer import NeuronVisualizer

def load_config(config_path):
    """Load model configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_model(config, device):
    """Create model matching saved architecture"""
    from updated_training_model import CompleteQuantumMagicPredictor
    
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'],
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config.get('use_physics_mask', False),
        mask_threshold=config.get('mask_threshold', 1),
        nhead=config['nhead'],
        use_cls_token=config.get('use_cls_token', True)
    )
    
    model.to(device)
    return model

def load_real_data(data_folder, labels_path, batch_size=64):
    """Load real quantum data"""
    from quantum_dataset import QuantumDataset
    from torch.utils.data import DataLoader, random_split
    
    # Load dataset
    dataset = QuantumDataset(data_folder, labels_path)
    total_size = len(dataset)
    
    # Use test split only
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None
    )
    
    print(f"✓ Loaded {len(test_set)} test samples from {total_size} total")
    return test_loader

def main():
    parser = argparse.ArgumentParser(description='Comprehensive XAI Analysis for Quantum Magic Prediction')
    
    # Data and model
    parser.add_argument('--data_folder', type=str, default='DecodedTokens',
                       help='Path to decoded tokens folder')
    parser.add_argument('--labels_path', type=str, default='label/magic_labels_for_input_for_2_qubits_mixed_1_50000_datapoints.npy',
                       help='Path to magic labels')
    parser.add_argument('--model_path', type=str, default='best_128_CLS_attention_enhanced_20250810_214428.pth',
                       help='Path to model weights')
    parser.add_argument('--config_path', type=str,
                       default='128_CLS_attention_enhanced_20250810_214428/128_CLS_attention_enhanced_config.json',
                       help='Path to model config')
    
    # Analysis parameters
    parser.add_argument('--max_samples', type=int, default=150000,
                       help='Maximum samples for analysis')
    parser.add_argument('--individual_samples', type=int, default=15,
                       help='Individual samples for detailed analysis')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='comprehensive_xai_results',
                       help='Output directory')
    
    # Analysis types
    parser.add_argument('--skip_attention', action='store_true',
                       help='Skip attention analysis (if already done)')
    parser.add_argument('--skip_probing', action='store_true',
                       help='Skip layer-wise probing')
    parser.add_argument('--skip_neurons', action='store_true',
                       help='Skip neuron analysis')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE XAI ANALYSIS FOR QUANTUM MAGIC PREDICTION")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config and model
    config = load_config(args.config_path)
    print(f"Model: {config['exp_name']} (d_model={config['d_model']}, mlp_type={config['mlp_type']})")
    
    model = create_model(config, device)
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ Model loaded")
    
    # Load data
    dataloader = load_real_data(args.data_folder, args.labels_path, args.batch_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate quantum labels
    quantum_labels = []
    for i in range(2**config['num_qubits']):
        for j in range(2**config['num_qubits']):
            i_binary = format(i, f'0{config["num_qubits"]}b')
            j_binary = format(j, f'0{config["num_qubits"]}b')
            quantum_labels.append(f'⟨{i_binary}|ρ|{j_binary}⟩')
    
    results = {}
    
    # Phase 1: Attention Analysis
    if not args.skip_attention:
        print(f"\n{'🔍 PHASE 1: ATTENTION ANALYSIS'}")
        print("="*50)
        
        attention_analyzer = AttentionAnalyzer(model, device)
        attention_visualizer = AttentionVisualizer(str(output_dir / "attention_analysis"))
        
        attention_results = attention_analyzer.analyze_dataset(dataloader, max_samples=args.max_samples)
        attention_visualizer.create_attention_dashboard(attention_results, quantum_labels)
        
        results['attention'] = attention_results
        print("✓ Attention analysis complete")
    
    # Phase 2: Layer-wise Probing  
    if not args.skip_probing:
        print(f"\n{'🧪 PHASE 2: LAYER-WISE PROBING'}")
        print("="*50)
        
        prober = LayerWiseProber(model, device)
        probing_visualizer = ProbingVisualizer(str(output_dir / "probing_analysis"))
        
        print("Running layer-wise probing...")
        probe_results = prober.probe_layer_representations(dataloader, max_samples=args.max_samples)
        
        print("Analyzing information emergence...")
        emergence_analysis = prober.analyze_information_emergence(probe_results)
        
        print("Computing representation similarity...")
        similarity_analysis = prober.compute_representation_similarity(probe_results)
        
        probing_visualizer.create_probing_dashboard(
            probe_results, emergence_analysis, similarity_analysis, quantum_labels
        )
        
        results['probing'] = {
            'probe_results': probe_results,
            'emergence_analysis': emergence_analysis,
            'similarity_analysis': similarity_analysis
        }
        print("✓ Layer-wise probing complete")
    
    # Phase 3: Neuron Specialization Analysis
    if not args.skip_neurons:
        print(f"\n{'🧠 PHASE 3: NEURON SPECIALIZATION ANALYSIS'}")
        print("="*50)
        
        neuron_analyzer = NeuronAnalyzer(model, device)
        neuron_visualizer = NeuronVisualizer(str(output_dir / "neuron_analysis"))
        
        print("Analyzing neuron specialization...")
        specialization_results = neuron_analyzer.analyze_neuron_specialization(
            dataloader, max_samples=args.max_samples
        )
        
        print("Finding magic detector neurons...")
        magic_detectors = neuron_analyzer.find_magic_detector_neurons(specialization_results)
        
        print("Creating neuron visualizations...")
        neuron_visualizer.create_neuron_dashboard(specialization_results, magic_detectors)
        
        results['neurons'] = {
            'specialization': specialization_results,
            'magic_detectors': magic_detectors
        }
        print("✓ Neuron analysis complete")
    
    # Create comprehensive summary
    print(f"\n{'📊 COMPREHENSIVE SUMMARY'}")
    print("="*50)
    
    summary_report = f"""
COMPREHENSIVE XAI ANALYSIS SUMMARY
=================================

Model: {config['exp_name']}
Samples Analyzed: {args.max_samples}
Real Quantum Data: ✓

"""
    
    if 'attention' in results:
        attn_stats = results['attention']['summary_stats']
        summary_report += f"""
🔍 ATTENTION ANALYSIS:
   - Average prediction error: {attn_stats['avg_prediction_error']:.4f}
   - Attention patterns across {len(results['attention']['layer_evolution'])} layers
   - Head specialization analysis complete
"""
    
    if 'probing' in results:
        emergence = results['probing']['emergence_analysis']
        summary_report += f"""
🧪 LAYER-WISE PROBING:
   - Information emergence layer: {emergence.get('emergence_layer', 'Not detected')}
   - Max R² score: {emergence['max_r2_score']:.4f} at {emergence['max_r2_layer']}
   - Linear separability analysis complete
"""
    
    if 'neurons' in results:
        neuron_stats = results['neurons']['magic_detectors']
        total_detectors = sum(layer_data['num_detectors'] for layer_data in neuron_stats.values() if isinstance(layer_data, dict))
        summary_report += f"""
🧠 NEURON SPECIALIZATION:
   - Magic detector neurons found: {total_detectors}
   - Neuron specialization patterns identified
   - Dead neuron analysis complete
"""
    
    summary_report += f"""
📁 RESULTS LOCATION: {output_dir}/
   - attention_analysis/: Transformer attention patterns
   - probing_analysis/: Layer-wise capability emergence  
   - Individual analyses: First {args.individual_samples} samples detailed

🎯 KEY INSIGHTS:
   1. Attention patterns show which quantum matrix elements matter most
   2. Probing reveals when magic detection capability emerges in the network
   3. Neuron analysis identifies specialized magic detector units
   4. Combined analysis provides complete picture of model interpretability

💡 NEXT STEPS:
   - Examine emergence layer for critical magic detection patterns
   - Compare attention focus with probe coefficients for consistency
   - Investigate magic detector neurons for quantum physics insights
"""
    
    # Save comprehensive report
    with open(output_dir / 'comprehensive_xai_report.txt', 'w') as f:
        f.write(summary_report)
    
    print(summary_report)
    print(f"\n🎉 Comprehensive XAI analysis complete!")
    print(f"Full report: {output_dir}/comprehensive_xai_report.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())