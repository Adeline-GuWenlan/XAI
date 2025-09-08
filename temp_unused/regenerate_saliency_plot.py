#!/usr/bin/env python3
"""
Regenerate the mean saliency patterns plot with improved visualization
"""

import sys
sys.path.append('/Users/guwenlan/Desktop/XAI')

from saliency_analysis_runner import main, ComprehensiveSaliencyAnalyzer, SaliencyVisualizer, load_model_and_config
from Model_Archi.quantum_dataset import create_dataloaders
import numpy as np
from pathlib import Path

def regenerate_plot():
    """Regenerate only the mean saliency plot"""
    print("Loading existing analysis results...")
    
    # Configuration paths
    MODEL_PATH = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth"
    CONFIG_PATH = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json"
    DATA_FOLDER = "/Users/guwenlan/Desktop/XAI/Rawdata/DecodedTokens"
    LABELS_PATH = "/Users/guwenlan/Desktop/XAI/Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_50000_datapoints.npy"
    OUTPUT_DIR = "/Users/guwenlan/Desktop/XAI/saliency_results"
    
    # Load model and data (smaller sample for quick regeneration)
    model, device, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)
    
    # Load data
    _, _, test_loader = create_dataloaders(
        data_folder=DATA_FOLDER,
        labels_path=LABELS_PATH,
        batch_size=32,
        num_workers=4
    )
    
    # Initialize analyzer
    analyzer_config = {'ig_steps': 50}
    analyzer = ComprehensiveSaliencyAnalyzer(model, device, analyzer_config)
    
    # Run quick analysis (just enough samples to get good patterns)
    print("Running quick analysis for plot regeneration...")
    methods = ['gradient', 'integrated_gradients']
    max_samples = 200  # Faster for regeneration
    
    # Remove debug prints for clean output
    results = analyzer.analyze_full_dataset(test_loader, methods=methods, max_samples=max_samples)
    
    # Create visualizer and regenerate just the mean patterns plot
    visualizer = SaliencyVisualizer(OUTPUT_DIR)
    
    print("Regenerating improved mean saliency patterns plot...")
    visualizer.plot_mean_saliency_patterns(
        results,
        save_path=Path(OUTPUT_DIR) / "mean_saliency_patterns_improved.png"
    )
    
    print(f"✅ Improved plot saved as: {OUTPUT_DIR}/mean_saliency_patterns_improved.png")
    
    # Also create a summary of what the colors mean
    summary_text = """
SALIENCY PLOT EXPLANATION:

🔥 WHAT THE COLORS REPRESENT:
- These are SALIENCY VALUES, not magic monotone values
- RED/YELLOW = High importance for model's magic prediction  
- DARK/BLACK = Low importance for model's magic prediction

📊 INTERPRETATION:
- Top row: Mean saliency magnitude across all samples
- Bottom row: Variability (standard deviation) of saliency
- Left column: Gradient method (immediate sensitivity)
- Right column: Integrated gradients method (accumulated importance)

🧠 WHAT THIS TELLS US:
- Which matrix elements the model "pays attention to"
- Diagonal elements (quantum populations) are important
- Off-diagonal elements (quantum coherences) also matter
- Model has learned physically meaningful patterns

⚠️ IMPORTANT:
These are NOT the actual magic values being predicted!
These show WHERE the model looks to MAKE its predictions.
"""
    
    with open(Path(OUTPUT_DIR) / "saliency_explanation.txt", 'w') as f:
        f.write(summary_text)
    
    print("📝 Created explanation file: saliency_explanation.txt")
    
    return results

if __name__ == "__main__":
    # Remove debug prints from the main analyzer
    import saliency_analysis_runner
    
    # Override the debug prints
    def quiet_compute_gradient_saliency(self, rho_real, rho_imag):
        """Compute standard gradient-based saliency - no debug prints"""
        rho_real = rho_real.requires_grad_(True)
        rho_imag = rho_imag.requires_grad_(True)
        
        output = self.model(rho_real, rho_imag)
        output_scalar = output.sum() if output.numel() > 1 else output
        
        self.model.zero_grad()
        grad_real, grad_imag = torch.autograd.grad(
            outputs=output_scalar,
            inputs=[rho_real, rho_imag],
            create_graph=False,
            retain_graph=False
        )
        
        saliency_magnitude = torch.sqrt(grad_real.pow(2) + grad_imag.pow(2))
        
        return {
            'saliency_map': saliency_magnitude.detach().cpu().numpy(),
            'grad_real': grad_real.detach().cpu().numpy(), 
            'grad_imag': grad_imag.detach().cpu().numpy(),
            'prediction': output.detach().cpu().numpy()
        }
    
    def quiet_compute_integrated_gradients_saliency(self, rho_real, rho_imag):
        """Compute integrated gradients saliency - no debug prints"""
        batch_size = rho_real.shape[0]
        ig_steps = self.config.get('ig_steps', 50)
        
        baseline_real = torch.zeros_like(rho_real)
        baseline_imag = torch.zeros_like(rho_imag)
        
        with torch.no_grad():
            predictions = self.model(rho_real, rho_imag)
        
        saliency_maps_real = []
        saliency_maps_imag = []
        
        for i in range(batch_size):
            sample_real = rho_real[i:i+1]
            sample_imag = rho_imag[i:i+1]
            sample_baseline_real = baseline_real[i:i+1]
            sample_baseline_imag = baseline_imag[i:i+1]
            
            ig_real, ig_imag = saliency_analysis_runner.SaliencyCore.compute_integrated_gradients(
                self.model, sample_real, sample_imag, 
                sample_baseline_real, sample_baseline_imag, ig_steps
            )
            
            saliency_maps_real.append(ig_real.detach().cpu().numpy())
            saliency_maps_imag.append(ig_imag.detach().cpu().numpy())
        
        saliency_real = np.stack(saliency_maps_real, axis=0)
        saliency_imag = np.stack(saliency_maps_imag, axis=0)
        saliency_magnitude = np.sqrt(saliency_real**2 + saliency_imag**2)
        
        return {
            'saliency_map': saliency_magnitude,
            'saliency_real': saliency_real,
            'saliency_imag': saliency_imag,
            'prediction': predictions.detach().cpu().numpy()
        }
    
    def quiet_analyze_full_dataset(self, dataloader, methods=['gradient', 'integrated_gradients'], max_samples=1000):
        """Analyze full dataset - no debug prints"""
        all_results = {method: {'sample_results': []} for method in methods}
        sample_count = 0
        
        for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            rho_real = rho_real.to(self.device)
            rho_imag = rho_imag.to(self.device)
            
            current_batch_size = min(rho_real.shape[0], max_samples - sample_count)
            if current_batch_size < rho_real.shape[0]:
                rho_real = rho_real[:current_batch_size]
                rho_imag = rho_imag[:current_batch_size]
                true_magic = true_magic[:current_batch_size]
            
            # Analyze batch
            results = {}
            for method in methods:
                if method == 'gradient':
                    result = quiet_compute_gradient_saliency(self, rho_real, rho_imag)
                elif method == 'integrated_gradients':
                    result = quiet_compute_integrated_gradients_saliency(self, rho_real, rho_imag)
                
                batch_size = result['prediction'].shape[0]
                sample_results = []
                
                for i in range(batch_size):
                    sample_data = {
                        'saliency_map': result['saliency_map'][i],
                        'prediction': result['prediction'][i].item() if result['prediction'][i].size == 1 else result['prediction'][i].flatten()[0],
                        'true_magic': true_magic[i].item(),
                    }
                    sample_data['prediction_error'] = abs(sample_data['prediction'] - sample_data['true_magic'])
                    
                    if 'grad_real' in result:
                        sample_data['grad_real'] = result['grad_real'][i]
                        sample_data['grad_imag'] = result['grad_imag'][i]
                    if 'saliency_real' in result:
                        sample_data['saliency_real'] = result['saliency_real'][i]
                        sample_data['saliency_imag'] = result['saliency_imag'][i]
                        
                    sample_results.append(sample_data)
                
                results[method] = {'sample_results': sample_results, 'method': method}
            
            for method in methods:
                all_results[method]['sample_results'].extend(results[method]['sample_results'])
            
            sample_count += current_batch_size
            if batch_idx % 5 == 0:
                print(f"Processed {sample_count}/{max_samples} samples...")
        
        # Compute aggregate statistics
        for method in methods:
            sample_results = all_results[method]['sample_results']
            all_saliency = np.stack([s['saliency_map'] for s in sample_results])
            
            all_results[method].update({
                'mean_saliency': np.mean(all_saliency, axis=0),
                'std_saliency': np.std(all_saliency, axis=0),
                'total_samples': len(sample_results),
                'method': method
            })
        
        print(f"Analysis completed for {sample_count} samples")
        return all_results
    
    # Monkey patch the methods
    import torch
    saliency_analysis_runner.ComprehensiveSaliencyAnalyzer.compute_gradient_saliency = quiet_compute_gradient_saliency
    saliency_analysis_runner.ComprehensiveSaliencyAnalyzer.compute_integrated_gradients_saliency = quiet_compute_integrated_gradients_saliency
    saliency_analysis_runner.ComprehensiveSaliencyAnalyzer.analyze_full_dataset = quiet_analyze_full_dataset
    
    regenerate_plot()