#!/usr/bin/env python3
"""
Comprehensive Saliency Analysis Script for Quantum Magic Detection Model
Implements normal saliency, integrated gradients, and mean saliency analysis with visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add current directory to path to import local modules
sys.path.append('/Users/guwenlan/Desktop/XAI')
sys.path.append('/Users/guwenlan/Desktop/XAI/Model_Archi')
sys.path.append('/Users/guwenlan/Desktop/XAI/temp_unused/Saliency')

# Import necessary modules
from Model_Archi.quantum_dataset import create_dataloaders
from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor

class SaliencyCore:
    """Core saliency computation functions"""
    
    @staticmethod
    def interpolate_inputs(baseline, inputs, steps):
        """Interpolate between baseline and inputs for integrated gradients"""
        alphas = torch.linspace(0, 1, steps, device=inputs.device)
        alphas = alphas.view(-1, *([1] * (len(inputs.shape))))
        return baseline + alphas * (inputs - baseline)
    
    @staticmethod
    def compute_integrated_gradients(model, inputs_real, inputs_imag, baseline_real, baseline_imag, steps=50):
        """Compute integrated gradients for complex-valued inputs"""
        # Ensure inputs are single samples [D, D] not batched [1, D, D]
        if inputs_real.dim() == 3:
            inputs_real = inputs_real.squeeze(0)
            inputs_imag = inputs_imag.squeeze(0)
            baseline_real = baseline_real.squeeze(0)
            baseline_imag = baseline_imag.squeeze(0)
        
        # Interpolate both real and imaginary parts
        interpolated_real = SaliencyCore.interpolate_inputs(baseline_real, inputs_real, steps)
        interpolated_imag = SaliencyCore.interpolate_inputs(baseline_imag, inputs_imag, steps)
        
        gradients_real = []
        gradients_imag = []
        
        for i in range(steps):
            # Get interpolated inputs for this step - add batch dimension
            inp_real = interpolated_real[i].unsqueeze(0).requires_grad_(True)
            inp_imag = interpolated_imag[i].unsqueeze(0).requires_grad_(True)
            
            # Forward pass
            output = model(inp_real, inp_imag)
            output_scalar = output.sum() if output.numel() > 1 else output
            
            # Backward pass
            model.zero_grad()
            grad_real, grad_imag = torch.autograd.grad(
                outputs=output_scalar, 
                inputs=[inp_real, inp_imag], 
                create_graph=False, 
                retain_graph=False
            )
            
            gradients_real.append(grad_real.squeeze(0))
            gradients_imag.append(grad_imag.squeeze(0))
        
        # Average gradients
        avg_grad_real = torch.stack(gradients_real).mean(dim=0)
        avg_grad_imag = torch.stack(gradients_imag).mean(dim=0)
        
        # Compute integrated gradients
        ig_real = (inputs_real - baseline_real) * avg_grad_real
        ig_imag = (inputs_imag - baseline_imag) * avg_grad_imag
        
        return ig_real, ig_imag

class ComprehensiveSaliencyAnalyzer:
    """Main saliency analyzer with all methods"""
    
    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path("saliency_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def compute_gradient_saliency(self, rho_real, rho_imag):
        """Compute standard gradient-based saliency"""
        rho_real = rho_real.requires_grad_(True)
        rho_imag = rho_imag.requires_grad_(True)
        
        # Forward pass
        output = self.model(rho_real, rho_imag)
        output_scalar = output.sum() if output.numel() > 1 else output
        
        # Compute gradients
        self.model.zero_grad()
        grad_real, grad_imag = torch.autograd.grad(
            outputs=output_scalar,
            inputs=[rho_real, rho_imag],
            create_graph=False,
            retain_graph=False
        )
        
        # Compute magnitude
        saliency_magnitude = torch.sqrt(grad_real.pow(2) + grad_imag.pow(2))
        
        # Debug: check shapes
        print(f"Debug gradient: grad_real shape: {grad_real.shape}, saliency shape: {saliency_magnitude.shape}")
        
        return {
            'saliency_map': saliency_magnitude.detach().cpu().numpy(),
            'grad_real': grad_real.detach().cpu().numpy(), 
            'grad_imag': grad_imag.detach().cpu().numpy(),
            'prediction': output.detach().cpu().numpy()
        }
    
    def compute_integrated_gradients_saliency(self, rho_real, rho_imag):
        """Compute integrated gradients saliency"""
        batch_size = rho_real.shape[0]
        ig_steps = self.config.get('ig_steps', 50)
        
        # Use zero baselines
        baseline_real = torch.zeros_like(rho_real)
        baseline_imag = torch.zeros_like(rho_imag)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(rho_real, rho_imag)
        
        # Process each sample
        saliency_maps_real = []
        saliency_maps_imag = []
        
        for i in range(batch_size):
            sample_real = rho_real[i:i+1]
            sample_imag = rho_imag[i:i+1]
            sample_baseline_real = baseline_real[i:i+1]
            sample_baseline_imag = baseline_imag[i:i+1]
            
            # Compute IG for this sample
            ig_real, ig_imag = SaliencyCore.compute_integrated_gradients(
                self.model, sample_real, sample_imag, 
                sample_baseline_real, sample_baseline_imag, ig_steps
            )
            
            saliency_maps_real.append(ig_real.detach().cpu().numpy())
            saliency_maps_imag.append(ig_imag.detach().cpu().numpy())
        
        # Stack results - ensure proper shape preservation
        saliency_real = np.stack(saliency_maps_real, axis=0)
        saliency_imag = np.stack(saliency_maps_imag, axis=0)
        saliency_magnitude = np.sqrt(saliency_real**2 + saliency_imag**2)
        
        # Debug: print shapes
        print(f"Debug: saliency_real shape: {saliency_real.shape}")
        print(f"Debug: saliency_magnitude shape: {saliency_magnitude.shape}")
        
        return {
            'saliency_map': saliency_magnitude,
            'saliency_real': saliency_real,
            'saliency_imag': saliency_imag,
            'prediction': predictions.detach().cpu().numpy()
        }
    
    def analyze_batch(self, rho_real, rho_imag, true_magic, methods=['gradient', 'integrated_gradients']):
        """Analyze a single batch with multiple methods"""
        results = {}
        
        for method in methods:
            print(f"Computing {method} saliency...")
            
            if method == 'gradient':
                result = self.compute_gradient_saliency(rho_real, rho_imag)
            elif method == 'integrated_gradients':
                result = self.compute_integrated_gradients_saliency(rho_real, rho_imag)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            # Add metadata
            batch_size = result['prediction'].shape[0]
            sample_results = []
            
            for i in range(batch_size):
                sample_data = {
                    'saliency_map': result['saliency_map'][i],
                    'prediction': result['prediction'][i].item() if result['prediction'][i].size == 1 else result['prediction'][i].flatten()[0],
                    'true_magic': true_magic[i].item(),
                }
                
                # Add prediction error
                sample_data['prediction_error'] = abs(sample_data['prediction'] - sample_data['true_magic'])
                
                # Add method-specific data
                if 'grad_real' in result:
                    sample_data['grad_real'] = result['grad_real'][i]
                    sample_data['grad_imag'] = result['grad_imag'][i]
                if 'saliency_real' in result:
                    sample_data['saliency_real'] = result['saliency_real'][i]
                    sample_data['saliency_imag'] = result['saliency_imag'][i]
                    
                sample_results.append(sample_data)
            
            results[method] = {
                'sample_results': sample_results,
                'method': method
            }
        
        return results
    
    def analyze_full_dataset(self, dataloader, methods=['gradient', 'integrated_gradients'], max_samples=1000):
        """Analyze full dataset and compute statistics"""
        print(f"Analyzing dataset with methods: {methods}")
        print(f"Maximum samples: {max_samples}")
        
        all_results = {method: {'sample_results': []} for method in methods}
        sample_count = 0
        
        for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            # Move to device  
            rho_real = rho_real.to(self.device)
            rho_imag = rho_imag.to(self.device)
            
            # Limit batch size if needed
            current_batch_size = min(rho_real.shape[0], max_samples - sample_count)
            if current_batch_size < rho_real.shape[0]:
                rho_real = rho_real[:current_batch_size]
                rho_imag = rho_imag[:current_batch_size]
                true_magic = true_magic[:current_batch_size]
            
            # Analyze batch
            batch_results = self.analyze_batch(rho_real, rho_imag, true_magic, methods)
            
            # Accumulate results
            for method in methods:
                all_results[method]['sample_results'].extend(batch_results[method]['sample_results'])
            
            sample_count += current_batch_size
            print(f"Processed {sample_count}/{max_samples} samples")
        
        # Compute aggregate statistics
        for method in methods:
            sample_results = all_results[method]['sample_results']
            all_saliency = np.stack([s['saliency_map'] for s in sample_results])
            
            print(f"Debug aggregation: method={method}, all_saliency shape={all_saliency.shape}")
            mean_sal = np.mean(all_saliency, axis=0)
            print(f"Debug aggregation: mean_saliency shape={mean_sal.shape}")
            
            all_results[method].update({
                'mean_saliency': mean_sal,
                'std_saliency': np.std(all_saliency, axis=0),
                'total_samples': len(sample_results),
                'method': method
            })
        
        print(f"Analysis completed for {sample_count} samples")
        return all_results

class SaliencyVisualizer:
    """Visualization tools for saliency analysis"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_saliency_map(self, saliency_map, title, save_path=None, vmax=None):
        """Plot a single saliency map"""
        plt.figure(figsize=(8, 6))
        
        if vmax is None:
            vmax = np.percentile(saliency_map, 95)  # Use 95th percentile to avoid outliers
            
        plt.imshow(saliency_map, cmap='hot', interpolation='nearest', vmax=vmax)
        plt.colorbar(label='Saliency')
        plt.title(title)
        plt.xlabel('Matrix Column')
        plt.ylabel('Matrix Row') 
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_method_comparison(self, results, sample_idx, true_magic, prediction, save_path=None):
        """Compare saliency maps from different methods for a single sample"""
        methods = list(results.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]
            
        # Find common scale
        all_saliency = []
        for method in methods:
            all_saliency.append(results[method]['sample_results'][sample_idx]['saliency_map'])
        
        vmax = np.percentile(np.concatenate([s.flatten() for s in all_saliency]), 95)
        
        for idx, method in enumerate(methods):
            sample_result = results[method]['sample_results'][sample_idx] 
            saliency_map = sample_result['saliency_map']
            
            im = axes[idx].imshow(saliency_map, cmap='hot', interpolation='nearest', vmax=vmax)
            axes[idx].set_title(f'{method.title()}\nSaliency')
            axes[idx].set_xlabel('Matrix Column')
            if idx == 0:
                axes[idx].set_ylabel('Matrix Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, orientation='horizontal', 
                           pad=0.1, aspect=30, shrink=0.8)
        cbar.set_label('Saliency Magnitude')
        
        plt.suptitle(f'Sample {sample_idx}: True Magic = {true_magic:.4f}, '
                    f'Predicted = {prediction:.4f}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_mean_saliency_patterns(self, results, save_path=None):
        """Plot mean saliency patterns across all methods"""
        methods = list(results.keys())
        n_methods = len(methods)
        
        # Create figure with proper spacing for colorbars
        fig = plt.figure(figsize=(6*n_methods, 10))
        
        # Find common scales
        all_means = [results[method]['mean_saliency'] for method in methods]
        all_stds = [results[method]['std_saliency'] for method in methods]
        
        mean_vmax = np.percentile(np.concatenate([m.flatten() for m in all_means]), 95)
        std_vmax = np.percentile(np.concatenate([s.flatten() for s in all_stds]), 95)
        
        # Create subplot grid with space for colorbars and title
        gs = fig.add_gridspec(2, n_methods, hspace=0.5, wspace=0.35, 
                             left=0.08, right=0.92, top=0.85, bottom=0.18)
        
        for idx, method in enumerate(methods):
            mean_saliency = results[method]['mean_saliency']
            std_saliency = results[method]['std_saliency']
            
            # Mean saliency subplot
            ax1 = fig.add_subplot(gs[0, idx])
            im1 = ax1.imshow(mean_saliency, cmap='hot', interpolation='nearest', vmax=mean_vmax)
            ax1.set_title(f'{method.replace("_", " ").title()}\nMean Saliency Magnitude', fontsize=12, pad=15)
            ax1.set_xlabel('Density Matrix Column Index', fontsize=10)
            if idx == 0:
                ax1.set_ylabel('Density Matrix Row Index', fontsize=10)
            
            # Add individual colorbar for mean saliency
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20, pad=0.05)
            cbar1.set_label('Input Importance\n(Higher = More Important for Magic Prediction)', fontsize=9)
            
            # Standard deviation subplot
            ax2 = fig.add_subplot(gs[1, idx])
            im2 = ax2.imshow(std_saliency, cmap='viridis', interpolation='nearest', vmax=std_vmax)
            ax2.set_title(f'{method.replace("_", " ").title()}\nSaliency Variability', fontsize=12, pad=15)
            ax2.set_xlabel('Density Matrix Column Index', fontsize=10)
            if idx == 0:
                ax2.set_ylabel('Density Matrix Row Index', fontsize=10)
            
            # Add individual colorbar for std deviation
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20, pad=0.05)
            cbar2.set_label('Saliency Consistency\n(Lower = More Consistent Across Samples)', fontsize=9)
        
        # Add main title with proper positioning
        fig.suptitle('Quantum Density Matrix Saliency Analysis\n' + 
                    'Model Attention Patterns for Magic Monotone Prediction', 
                    fontsize=14, y=0.95)
        
        # Add explanation text at bottom
        explanation = ("Saliency maps show which density matrix elements are most important for the model's magic predictions.\n"
                      "Hot colors (red/yellow) indicate high importance. These are NOT magic values, but model attention weights.")
        fig.text(0.5, 0.05, explanation, ha='center', fontsize=10, style='italic', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_prediction_vs_saliency(self, results, save_path=None):
        """Plot prediction accuracy vs saliency magnitude"""
        methods = list(results.keys())
        
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            sample_results = results[method]['sample_results']
            
            errors = [s['prediction_error'] for s in sample_results]
            saliency_magnitudes = [np.mean(s['saliency_map']) for s in sample_results]
            
            axes[idx].scatter(saliency_magnitudes, errors, alpha=0.6, s=20)
            axes[idx].set_xlabel('Mean Saliency Magnitude')
            axes[idx].set_ylabel('Prediction Error')
            axes[idx].set_title(f'{method.title()}\nSaliency vs Error')
            axes[idx].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(saliency_magnitudes, errors)[0, 1]
            axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                          transform=axes[idx].transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()

def load_model_and_config(model_path, config_path):
    """Load model with configuration"""
    import json
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        d_model=config['d_model'], 
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config['use_physics_mask'],
        mask_threshold=config.get('mask_threshold', 1),
        nhead=config.get('nhead', 8),
        use_cls_token=config.get('use_cls_token', True)
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model config: {config}")
    
    return model, device, config

def main():
    """Main analysis function"""
    print("=" * 60)
    print("COMPREHENSIVE SALIENCY ANALYSIS")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth"
    CONFIG_PATH = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json"
    DATA_FOLDER = "/Users/guwenlan/Desktop/XAI/Rawdata/DecodedTokens"
    LABELS_PATH = "/Users/guwenlan/Desktop/XAI/Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_50000_datapoints.npy"
    OUTPUT_DIR = "/Users/guwenlan/Desktop/XAI/saliency_results"
    
    # Load model
    print("Loading model and configuration...")
    model, device, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)
    
    # Load data
    print("Loading dataset...")
    _, _, test_loader = create_dataloaders(
        data_folder=DATA_FOLDER,
        labels_path=LABELS_PATH,
        batch_size=32,  # Smaller batch size for memory efficiency
        num_workers=4
    )
    
    # Initialize analyzer
    analyzer_config = {
        'ig_steps': 50
    }
    analyzer = ComprehensiveSaliencyAnalyzer(model, device, analyzer_config)
    visualizer = SaliencyVisualizer(OUTPUT_DIR)
    
    # Run analysis
    methods = ['gradient', 'integrated_gradients'] 
    max_samples = 500  # Full analysis
    
    print(f"Running saliency analysis...")
    print(f"Methods: {methods}")
    print(f"Max samples: {max_samples}")
    
    results = analyzer.analyze_full_dataset(test_loader, methods=methods, max_samples=max_samples)
    
    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    # 1. Mean saliency patterns
    print("1. Plotting mean saliency patterns...")
    visualizer.plot_mean_saliency_patterns(
        results, 
        save_path=Path(OUTPUT_DIR) / "mean_saliency_patterns.png"
    )
    
    # 2. Individual sample comparisons (first 10 samples)
    print("2. Generating individual sample comparisons...")
    n_samples_to_show = min(10, len(results[methods[0]]['sample_results']))
    
    for i in range(n_samples_to_show):
        sample_result = results[methods[0]]['sample_results'][i]
        true_magic = sample_result['true_magic']
        prediction = sample_result['prediction']
        
        visualizer.plot_method_comparison(
            results, i, true_magic, prediction,
            save_path=Path(OUTPUT_DIR) / f"sample_{i:03d}_comparison.png"
        )
    
    # 3. Prediction accuracy vs saliency
    print("3. Plotting prediction accuracy vs saliency...")
    visualizer.plot_prediction_vs_saliency(
        results,
        save_path=Path(OUTPUT_DIR) / "prediction_vs_saliency.png"
    )
    
    # Print summary statistics
    print("\n" + "=" * 40)
    print("ANALYSIS SUMMARY")
    print("=" * 40)
    
    for method in methods:
        sample_results = results[method]['sample_results']
        errors = [r['prediction_error'] for r in sample_results]
        true_values = [r['true_magic'] for r in sample_results]
        predictions = [r['prediction'] for r in sample_results]
        saliency_means = [np.mean(r['saliency_map']) for r in sample_results]
        
        print(f"\n{method.upper()} RESULTS:")
        print(f"  Samples analyzed: {len(sample_results)}")
        print(f"  Mean prediction error: {np.mean(errors):.6f} ± {np.std(errors):.6f}")
        print(f"  True magic range: [{np.min(true_values):.4f}, {np.max(true_values):.4f}]")
        print(f"  Predicted range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        print(f"  Mean saliency magnitude: {np.mean(saliency_means):.6f} ± {np.std(saliency_means):.6f}")
        print(f"  Max saliency value: {np.max([np.max(r['saliency_map']) for r in sample_results]):.6f}")
    
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"📊 Generated visualizations:")
    print(f"   - mean_saliency_patterns.png")  
    print(f"   - sample_xxx_comparison.png (x{n_samples_to_show})")
    print(f"   - prediction_vs_saliency.png")
    
    return results

if __name__ == "__main__":
    results = main()