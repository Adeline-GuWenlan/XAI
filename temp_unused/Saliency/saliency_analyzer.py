import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from saliency_core import compute_integrated_gradients

class SaliencyAnalyzer:
    """Analyzer for computing saliency maps using various gradient-based methods"""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict = None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.eval()
        
    def compute_gradient_saliency(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict:
        """
        Compute gradient-based saliency map
        
        Args:
            rho_real, rho_imag: Input density matrix components [B, D, D]
            
        Returns:
            Dict containing saliency maps and predictions
        """
        rho_real.requires_grad_(True)
        rho_imag.requires_grad_(True)
        
        # Forward pass
        output = self.model(rho_real, rho_imag)
        
        # Compute gradients
        self.model.zero_grad()
        grad_outputs = torch.ones_like(output)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=[rho_real, rho_imag],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )
        
        # Combine real and imaginary gradients
        grad_real, grad_imag = gradients
        
        # Compute magnitude of complex gradient
        saliency_map = torch.sqrt(grad_real.pow(2) + grad_imag.pow(2))
        
        return {
            'saliency_map': saliency_map.detach().cpu().numpy(),
            'grad_real': grad_real.detach().cpu().numpy(),
            'grad_imag': grad_imag.detach().cpu().numpy(),
            'prediction': output.detach().cpu().numpy()
        }
        
    def compute_integrated_gradients_saliency(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict:
        """
        Compute integrated gradients saliency map
        
        Args:
            rho_real, rho_imag: Input density matrix components [B, D, D]
            
        Returns:
            Dict containing saliency maps and predictions
        """
        batch_size = rho_real.shape[0]
        ig_steps = self.config.get('ig_steps', 50)
        
        # Compute baseline (zeros)
        baseline_real = torch.zeros_like(rho_real)
        baseline_imag = torch.zeros_like(rho_imag)
        
        # Get prediction for original input
        with torch.no_grad():
            prediction = self.model(rho_real, rho_imag)
        
        saliency_maps_real = []
        saliency_maps_imag = []
        
        # Ensure model is in training mode for gradient computation
        self.model.train()
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Single sample - clone to ensure proper gradient computation
            sample_real = rho_real[i:i+1].clone()
            sample_imag = rho_imag[i:i+1].clone()
            sample_baseline_real = baseline_real[i:i+1].clone()
            sample_baseline_imag = baseline_imag[i:i+1].clone()
            
            # Create fixed copies for the wrapper functions
            fixed_imag = sample_imag.clone().detach()
            fixed_real = sample_real.clone().detach()
            
            # For models with CLS pooling, we need to compute gradients w.r.t. the embedded inputs
            # The model adds CLS token internally, so we need a wrapper that computes gradients 
            # for the matrix elements only
            
            # Compute integrated gradients for real part
            def model_real_wrapper(x):
                # x is the variable input (real part)
                # fixed_imag is the fixed imaginary part
                self.model.zero_grad()
                output = self.model(x, fixed_imag)
                return output
            
            ig_real = compute_integrated_gradients(
                model=model_real_wrapper,
                inputs=sample_real,
                baseline=sample_baseline_real,
                steps=ig_steps,
                target=None
            )
            
            # Compute integrated gradients for imaginary part  
            def model_imag_wrapper(x):
                # x is the variable input (imaginary part)
                # fixed_real is the fixed real part
                self.model.zero_grad()
                output = self.model(fixed_real, x)
                return output
            
            ig_imag = compute_integrated_gradients(
                model=model_imag_wrapper,
                inputs=sample_imag,
                baseline=sample_baseline_imag,
                steps=ig_steps,
                target=None
            )
            
            saliency_maps_real.append(ig_real.cpu().numpy())
            saliency_maps_imag.append(ig_imag.cpu().numpy())
        
        # Stack results
        saliency_real = np.vstack(saliency_maps_real)
        saliency_imag = np.vstack(saliency_maps_imag)
        
        # Compute magnitude
        saliency_magnitude = np.sqrt(saliency_real**2 + saliency_imag**2)
        
        return {
            'saliency_map': saliency_magnitude,
            'saliency_real': saliency_real,
            'saliency_imag': saliency_imag,
            'prediction': prediction.detach().cpu().numpy()
        }
        
    def analyze_single_batch(self, rho_real: torch.Tensor, rho_imag: torch.Tensor, 
                           true_magic: torch.Tensor, methods: List[str]) -> Dict:
        """
        Analyze a single batch with specified methods
        
        Args:
            rho_real, rho_imag: Input density matrix components
            true_magic: True magic values
            methods: List of methods to use
            
        Returns:
            Dict with results for each method
        """
        results = {}
        
        for method in methods:
            if method == 'gradient':
                result = self.compute_gradient_saliency(rho_real, rho_imag)
            elif method == 'integrated_gradients':
                result = self.compute_integrated_gradients_saliency(rho_real, rho_imag)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add true values and compute errors
            batch_size = result['prediction'].shape[0]
            sample_results = []
            
            for i in range(batch_size):
                sample_data = {
                    'saliency_map': result['saliency_map'][i],
                    'prediction': result['prediction'][i].item(),
                    'true_magic': true_magic[i].item(),
                    'prediction_error': abs(result['prediction'][i].item() - true_magic[i].item())
                }
                
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
        
    def analyze_dataset(self, dataloader, methods: List[str] = ['integrated_gradients'], 
                       max_samples: int = 150000) -> Dict:
        """
        Analyze saliency patterns across a dataset
        
        Args:
            dataloader: DataLoader with quantum density matrices
            methods: List of saliency methods to use
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dict with comprehensive saliency analysis
        """
        print(f"Analyzing saliency patterns for up to {max_samples} samples...")
        print(f"Methods: {methods}")
        
        all_results = {method: {'sample_results': []} for method in methods}
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
                if sample_count >= max_samples:
                    break
                    
                # Move to device
                rho_real = rho_real.to(self.device)
                rho_imag = rho_imag.to(self.device)
                
                # Limit batch size if needed
                batch_size = min(rho_real.shape[0], max_samples - sample_count)
                if batch_size < rho_real.shape[0]:
                    rho_real = rho_real[:batch_size]
                    rho_imag = rho_imag[:batch_size]
                    true_magic = true_magic[:batch_size]
                
                # Analyze batch
                batch_results = self.analyze_single_batch(
                    rho_real, rho_imag, true_magic, methods
                )
                
                # Accumulate results
                for method in methods:
                    all_results[method]['sample_results'].extend(
                        batch_results[method]['sample_results']
                    )
                
                sample_count += batch_size
                if batch_idx % 10 == 0:
                    print(f"Processed {sample_count} samples...")
        
        # Compute aggregate statistics
        for method in methods:
            sample_results = all_results[method]['sample_results']
            
            # Compute aggregated saliency patterns
            all_saliency = np.stack([s['saliency_map'] for s in sample_results])
            
            all_results[method].update({
                'mean_saliency': np.mean(all_saliency, axis=0),
                'std_saliency': np.std(all_saliency, axis=0),
                'total_samples': len(sample_results),
                'method': method
            })
        
        print(f"Completed saliency analysis for {sample_count} samples")
        return all_results