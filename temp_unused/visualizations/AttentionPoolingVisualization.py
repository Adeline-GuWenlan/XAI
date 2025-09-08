#!/usr/bin/env python3
"""
AttentionPoolingVisualization.py

Specialized visualization tool for analyzing quantum magic prediction models that use
attention-based pooling instead of CLS tokens. This script focuses on visualizing
both the pooled representations and the attention mechanisms themselves.

Key Features:
- Extracts attention-pooled representations from models using PhysicsAwarePooling
- Visualizes attention weight patterns across quantum density matrices
- Supports PCA/UMAP/t-SNE analysis of pooled features
- Physics-informed analysis of attention patterns (diagonal vs off-diagonal)
- Organized output structure by model name

Author: Claude Code Assistant
Date: 2025-01-20
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from torch.utils.data import DataLoader
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import argparse
import os
import json
from pathlib import Path
import types
from typing import Dict, List, Tuple, Optional, Union


def _auto_tsne_perplexity(n_samples: int) -> int:
    """
    Automatically choose a safe perplexity for t-SNE given number of samples.
    
    Args:
        n_samples: Number of data samples
        
    Returns:
        Safe perplexity value ensuring 2 <= perplexity < n_samples
    """
    if n_samples <= 3:
        return 2
    # Aim for ~ n/3, clamp to [5, 30], then ensure < n
    p = min(30, max(5, (n_samples - 1) // 3))
    p = max(2, min(p, n_samples - 1))
    return int(p)


class AttentionPoolingVisualizationTool:
    """
    Comprehensive visualization tool for attention-based pooling in quantum magic prediction.
    
    This class provides analysis capabilities for models that use PhysicsAwarePooling
    with attention mechanisms instead of CLS tokens. It extracts and visualizes both
    the final pooled representations and the intermediate attention patterns.
    """
    
    def __init__(self, model, device='cpu', model_name="attention_model"):
        """
        Initialize the visualization tool.
        
        Args:
            model: Trained quantum magic prediction model with PhysicsAwarePooling
            device: PyTorch device ('cpu' or 'cuda')
            model_name: Name for organizing output files and folders
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.model_name = model_name
        
        # Verify model uses attention-based pooling
        self._verify_attention_pooling()
        
        # Add custom extraction methods to model
        self._add_extraction_methods()
        
        print(f"🔧 AttentionPoolingVisualizationTool initialized for model: {model_name}")
        print(f"🔧 Pooling type: {self.model.pooling_type}")
        print(f"🔧 Device: {device}")
    
    def _verify_attention_pooling(self):
        """
        Verify that the model uses attention-based or other non-CLS pooling.
        
        Raises:
            ValueError: If model uses CLS pooling (use Visualization.py instead)
        """
        if hasattr(self.model, 'pooling_type'):
            if self.model.pooling_type == "cls":
                raise ValueError(
                    "Model uses CLS pooling. Please use Visualization.py for CLS-based models."
                )
            print(f"✅ Detected pooling type: {self.model.pooling_type}")
        else:
            print("⚠️  Could not detect pooling type. Proceeding with attention extraction.")
    
    def _add_extraction_methods(self):
        """
        Add custom methods to the model for extracting attention-based representations.
        
        This dynamically adds methods to extract:
        1. Pooled representations (final features after attention pooling)
        2. Attention weights (if using attention pooling)
        3. Token-level representations (before pooling)
        """
        
        def extract_attention_representations(self, rho_real, rho_imag):
            """
            Extract attention-pooled representations and attention weights.
            
            Args:
                rho_real: Real part of quantum density matrices [B, D, D]
                rho_imag: Imaginary part of quantum density matrices [B, D, D]
                
            Returns:
                dict: Contains 'pooled_features', 'attention_weights', 'token_representations'
            """
            # Step 1: Embedding - convert matrices to token sequences
            matrix_tokens = self.embedding(rho_real, rho_imag)  # [B, D², d_model]
            B = matrix_tokens.shape[0]
            
            # Step 2: Add CLS token if present (even for non-CLS models, some have it)
            if hasattr(self, 'cls_token') and self.pooling_type != "cls":
                cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
                tokens = torch.cat([cls_tokens, matrix_tokens], dim=1)  # [B, D²+1, d_model]
            else:
                # For pure attention models, we still need to add a dummy token at position 0
                # to match the expected input format of PhysicsAwarePooling
                dummy_token = torch.zeros(B, 1, self.d_model, device=matrix_tokens.device)
                tokens = torch.cat([dummy_token, matrix_tokens], dim=1)  # [B, D²+1, d_model]
            
            # Step 3: Transformer processing
            for layer in self.encoder_layers:
                tokens = layer(tokens)  # [B, D²+1, d_model]
            
            # Step 4: Extract attention weights and pooled features
            attention_weights = None
            
            if self.pooling_type == "attention":
                # Extract matrix tokens (skip position 0)
                matrix_tokens_processed = tokens[:, 1:, :]  # [B, D², d_model]
                
                # Get attention weights before pooling
                attention_weights = torch.softmax(
                    self.physics_pooling.attention_weights(matrix_tokens_processed), 
                    dim=1
                )  # [B, D², 1]
                
                # Apply attention pooling
                pooled_features = self.physics_pooling(tokens)  # [B, d_model]
                
            else:
                # For other pooling types (mean, structured)
                pooled_features = self.physics_pooling(tokens)  # [B, d_model]
                attention_weights = None
            
            return {
                'pooled_features': pooled_features,      # [B, d_model] - main representations
                'attention_weights': attention_weights,  # [B, D², 1] or None
                'token_representations': tokens,         # [B, D²+1, d_model] - full sequence
                'matrix_tokens': tokens[:, 1:, :],      # [B, D², d_model] - matrix tokens only
            }
        
        # Bind the method to the model instance
        self.model.extract_attention_representations = types.MethodType(
            extract_attention_representations, self.model
        )
    
    def extract_representations(self, dataloader: DataLoader, max_samples: int = 150000) -> Dict:
        """
        Extract attention-based representations from the dataloader.
        
        Args:
            dataloader: DataLoader providing (rho_real, rho_imag, labels) batches
            max_samples: Maximum number of samples to process
            
        Returns:
            dict: Contains extracted representations, attention weights, and labels
        """
        print(f"🔍 Extracting attention-based representations (max_samples={max_samples})...")
        
        device = next(self.model.parameters()).device
        
        # Storage for extracted data
        extracted_data = {
            'pooled_features': [],      # Main representations for dimensionality reduction
            'attention_weights': [],    # Attention patterns (if available)
            'token_representations': [], # Full token sequences
            'matrix_tokens': [],        # Matrix tokens only
            'labels': []               # Ground truth magic values
        }
        
        self.model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Parse batch - expecting (rho_real, rho_imag, labels)
                if len(batch) != 3:
                    raise ValueError(f"Expected 3 items per batch, got {len(batch)}")
                
                rho_real, rho_imag, labels = batch
                
                # Move to device
                rho_real = rho_real.to(device)
                rho_imag = rho_imag.to(device)
                
                # Extract representations using our custom method
                representations = self.model.extract_attention_representations(rho_real, rho_imag)
                
                # Store data (move to CPU for memory efficiency)
                batch_size = rho_real.shape[0]
                extracted_data['pooled_features'].append(
                    representations['pooled_features'].detach().cpu()
                )
                extracted_data['token_representations'].append(
                    representations['token_representations'].detach().cpu()
                )
                extracted_data['matrix_tokens'].append(
                    representations['matrix_tokens'].detach().cpu()
                )
                
                if representations['attention_weights'] is not None:
                    extracted_data['attention_weights'].append(
                        representations['attention_weights'].detach().cpu()
                    )
                
                # Handle labels (flatten if needed)
                if labels.dim() > 1:
                    labels = labels.flatten()
                extracted_data['labels'].append(labels.detach().cpu())
                
                total_samples += batch_size
                if total_samples >= max_samples:
                    break
        
        # Concatenate all batches
        print("🔍 Concatenating extracted representations...")
        
        final_data = {}
        for key in extracted_data:
            if extracted_data[key]:  # If list is not empty
                final_data[key] = torch.cat(extracted_data[key], dim=0)
            else:
                final_data[key] = None
        
        # Print summary statistics
        print(f"✅ Extracted representations summary:")
        print(f"   📊 Total samples: {len(final_data['labels'])}")
        print(f"   📊 Pooled features shape: {final_data['pooled_features'].shape}")
        print(f"   📊 Token representations shape: {final_data['token_representations'].shape}")
        
        if final_data['attention_weights'] is not None:
            print(f"   📊 Attention weights shape: {final_data['attention_weights'].shape}")
        else:
            print(f"   📊 Attention weights: Not available (pooling_type != 'attention')")
        
        return final_data
    
    def visualize_pooled_space(self, representations: Dict, method: str = "umap", 
                              dim: int = 2, out_path: Optional[str] = None):
        """
        Visualize the pooled feature space using dimensionality reduction.
        
        Args:
            representations: Dictionary containing extracted representations
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            dim: Number of dimensions (2 or 3)
            out_path: Optional path to save the figure
        """
        # Extract pooled features and labels
        pooled_features = representations['pooled_features'].numpy()
        labels = representations['labels'].numpy()
        
        n_samples = len(pooled_features)
        n_features = pooled_features.shape[1]
        
        print(f"🔍 visualize_pooled_space: {method} {dim}D with {n_samples} samples, {n_features} features")
        
        # Validation checks
        if n_samples < dim:
            print(f"⚠️  Warning: Only {n_samples} samples available, cannot create {dim}D visualization")
            return
        
        if n_features < dim:
            print(f"⚠️  Warning: Only {n_features} features available, cannot create {dim}D visualization")
            return
        
        # Additional validation for t-SNE
        min_samples_needed = max(dim + 1, 10)
        if method.lower() == 'tsne':
            min_samples_needed = max(dim + 1, 30)
        
        if n_samples < min_samples_needed:
            print(f"⚠️  Warning: {method} needs at least {min_samples_needed} samples, got {n_samples}")
            return
        
        # Apply dimensionality reduction
        method = method.lower()
        if method == "pca":
            reducer = PCA(n_components=dim, random_state=42)
            embedding = reducer.fit_transform(pooled_features)
            
        elif method == "tsne":
            perplexity = _auto_tsne_perplexity(n_samples)
            reducer = TSNE(
                n_components=dim,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            )
            embedding = reducer.fit_transform(pooled_features)
            
        elif method == "umap":
            reducer = UMAP(
                n_components=dim,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            )
            embedding = reducer.fit_transform(pooled_features)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create visualization
        title = f"{method.upper()} ({dim}D) - Attention-Pooled Features"
        
        if dim == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=labels, cmap="viridis", alpha=0.7, s=30)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(title)
            colorbar = plt.colorbar(scatter, ax=ax)
            colorbar.set_label("Magic Value")
            
        else:  # 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=labels, cmap="viridis", alpha=0.7, s=20, depthshade=True)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.set_title(title)
            fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        
        # Save or show
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"📊 Saved {method} {dim}D visualization: {out_path}")
        else:
            plt.show()
    
    def visualize_attention_patterns(self, representations: Dict, out_path: Optional[str] = None):
        """
        Visualize attention weight patterns across quantum density matrices.
        
        This function creates heatmaps showing which matrix elements receive
        the most attention when making magic value predictions.
        
        Args:
            representations: Dictionary containing attention weights and labels
            out_path: Optional path to save the figure
        """
        attention_weights = representations['attention_weights']
        labels = representations['labels'].numpy()
        
        if attention_weights is None:
            print("⚠️  No attention weights available (model doesn't use attention pooling)")
            return
        
        attention_weights = attention_weights.numpy()  # [B, D², 1]
        attention_weights = attention_weights.squeeze(-1)  # [B, D²]
        
        # Determine matrix dimensions (assuming square matrices)
        D_squared = attention_weights.shape[1]
        D = int(np.sqrt(D_squared))
        
        if D * D != D_squared:
            print(f"⚠️  Warning: Cannot determine matrix dimensions from {D_squared} elements")
            return
        
        print(f"🔍 Visualizing attention patterns for {D}×{D} quantum matrices...")
        
        # Reshape attention weights to matrix form
        attention_matrices = attention_weights.reshape(-1, D, D)  # [B, D, D]
        
        # Split samples by magic value (high vs low)
        median_magic = np.median(labels)
        high_magic_mask = labels > median_magic
        low_magic_mask = labels <= median_magic
        
        # Average attention patterns for high/low magic samples
        high_magic_attention = attention_matrices[high_magic_mask].mean(axis=0)
        low_magic_attention = attention_matrices[low_magic_mask].mean(axis=0)
        overall_attention = attention_matrices.mean(axis=0)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Overall attention pattern
        im1 = axes[0, 0].imshow(overall_attention, cmap='Blues', interpolation='nearest')
        axes[0, 0].set_title('Overall Attention Pattern')
        axes[0, 0].set_xlabel('Matrix Column')
        axes[0, 0].set_ylabel('Matrix Row')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # High magic attention
        im2 = axes[0, 1].imshow(high_magic_attention, cmap='Reds', interpolation='nearest')
        axes[0, 1].set_title(f'High Magic States (>{median_magic:.3f})')
        axes[0, 1].set_xlabel('Matrix Column')
        axes[0, 1].set_ylabel('Matrix Row')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Low magic attention
        im3 = axes[0, 2].imshow(low_magic_attention, cmap='Greens', interpolation='nearest')
        axes[0, 2].set_title(f'Low Magic States (≤{median_magic:.3f})')
        axes[0, 2].set_xlabel('Matrix Column')
        axes[0, 2].set_ylabel('Matrix Row')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Difference between high and low magic attention
        attention_diff = high_magic_attention - low_magic_attention
        im4 = axes[1, 0].imshow(attention_diff, cmap='RdBu_r', interpolation='nearest',
                               vmin=-np.abs(attention_diff).max(), vmax=np.abs(attention_diff).max())
        axes[1, 0].set_title('Attention Difference (High - Low Magic)')
        axes[1, 0].set_xlabel('Matrix Column')
        axes[1, 0].set_ylabel('Matrix Row')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Element-wise attention distribution
        axes[1, 1].hist(overall_attention.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_xlabel('Attention Weight')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Attention Weights')
        
        # Physics-aware analysis: Diagonal vs Off-diagonal attention
        diag_indices = np.diag_indices(D)
        diag_attention = attention_matrices[:, diag_indices[0], diag_indices[1]]  # [B, D]
        offdiag_mask = ~np.eye(D, dtype=bool)
        offdiag_attention = attention_matrices[:, offdiag_mask]  # [B, D²-D]
        
        # Box plot comparing diagonal vs off-diagonal attention
        diag_mean = diag_attention.mean(axis=1)  # Average diagonal attention per sample
        offdiag_mean = offdiag_attention.mean(axis=1)  # Average off-diagonal attention per sample
        
        data_for_boxplot = [diag_mean, offdiag_mean]
        axes[1, 2].boxplot(data_for_boxplot, labels=['Diagonal', 'Off-diagonal'])
        axes[1, 2].set_ylabel('Average Attention Weight')
        axes[1, 2].set_title('Diagonal vs Off-diagonal Attention')
        
        plt.tight_layout()
        
        # Save or show
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"📊 Saved attention pattern analysis: {out_path}")
        else:
            plt.show()
    
    def analyze_feature_importance(self, representations: Dict, top_k: int = 10, 
                                 out_path: Optional[str] = None):
        """
        Analyze which dimensions of the pooled features are most important.
        
        Args:
            representations: Dictionary containing pooled features and labels
            top_k: Number of top dimensions to analyze
            out_path: Optional path to save the figure
            
        Returns:
            list: Top correlating dimensions and their correlation values
        """
        pooled_features = representations['pooled_features'].numpy()
        labels = representations['labels'].numpy()
        
        print(f"🔍 Analyzing feature importance for {pooled_features.shape[1]} dimensions...")
        
        # Ensure same number of samples
        min_samples = min(pooled_features.shape[0], labels.shape[0])
        pooled_features = pooled_features[:min_samples]
        labels = labels[:min_samples]
        
        # Compute correlations between each feature dimension and magic values
        correlations = []
        for i in range(pooled_features.shape[1]):
            try:
                corr = np.corrcoef(pooled_features[:, i], labels)[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, abs(corr)))
            except Exception as e:
                print(f"⚠️  Warning: Could not compute correlation for dimension {i}: {e}")
                continue
        
        if len(correlations) == 0:
            print("⚠️  Warning: No valid correlations found!")
            return []
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_correlations = correlations[:top_k]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of top correlations
        dims, corr_vals = zip(*top_correlations)
        bars = ax1.bar(range(len(dims)), corr_vals, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Absolute Correlation with Magic Value')
        ax1.set_title(f'Top {top_k} Most Important Pooled Feature Dimensions')
        ax1.set_xticks(range(len(dims)))
        ax1.set_xticklabels([f'Dim {d}' for d in dims], rotation=45)
        
        # Add correlation values on bars
        for bar, corr_val in zip(bars, corr_vals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{corr_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Scatter plot for the most important dimension
        best_dim, best_corr = top_correlations[0]
        scatter = ax2.scatter(pooled_features[:, best_dim], labels, 
                            alpha=0.6, c=labels, cmap='viridis', s=20)
        ax2.set_xlabel(f'Feature Dimension {best_dim}')
        ax2.set_ylabel('Magic Value')
        ax2.set_title(f'Most Predictive Dimension (r={best_corr:.3f})')
        plt.colorbar(scatter, ax=ax2)
        
        plt.tight_layout()
        
        # Save or show
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"📊 Saved feature importance analysis: {out_path}")
        else:
            plt.show()
        
        return top_correlations
    
    def generate_comprehensive_report(self, dataloader: DataLoader, output_dir: Optional[str] = None):
        """
        Generate a comprehensive analysis report for attention-based pooling models.
        
        Args:
            dataloader: DataLoader providing the test data
            output_dir: Directory to save all outputs (defaults to visualizations/{model_name}/)
            
        Returns:
            tuple: (representations, top_features) for further analysis
        """
        # Create output directory structure
        if output_dir is None:
            output_dir = Path("visualizations") / self.model_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Saving all outputs to: {output_dir}")
        
        # Step 1: Extract representations
        representations = self.extract_representations(dataloader, max_samples=150000)
        
        # Step 2: Generate all visualizations
        print("📊 Generating comprehensive visualizations...")
        
        # Pooled feature space visualizations (2D and 3D for each method)
        methods = ['pca', 'umap', 'tsne']
        dimensions = [2, 3]
        
        for method in methods:
            for dim in dimensions:
                out_path = output_dir / f"{method}_{dim}d_pooled_features.png"
                self.visualize_pooled_space(representations, method=method, dim=dim, out_path=str(out_path))
        
        # Attention pattern analysis (if available)
        if representations['attention_weights'] is not None:
            attention_path = output_dir / "attention_patterns.png"
            self.visualize_attention_patterns(representations, out_path=str(attention_path))
        
        # Feature importance analysis
        importance_path = output_dir / "feature_importance.png"
        top_features = self.analyze_feature_importance(representations, out_path=str(importance_path))
        
        # Step 3: Generate HTML report
        self._generate_html_report(representations, top_features, output_dir)
        
        print(f"✅ Comprehensive analysis complete!")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"📊 Generated visualizations:")
        for method in methods:
            for dim in dimensions:
                print(f"   - {method}_{dim}d_pooled_features.png")
        
        if representations['attention_weights'] is not None:
            print(f"   - attention_patterns.png")
        print(f"   - feature_importance.png")
        print(f"   - analysis_report.html")
        
        return representations, top_features
    
    def _generate_html_report(self, representations: Dict, top_features: List, output_dir: Path):
        """
        Generate a comprehensive HTML report with analysis summary.
        
        Args:
            representations: Extracted representation data
            top_features: List of top correlating features
            output_dir: Directory containing generated visualizations
        """
        report_path = output_dir / "analysis_report.html"
        
        # Compute summary statistics
        n_samples = len(representations['labels'])
        pooled_dim = representations['pooled_features'].shape[1]
        magic_values = representations['labels'].numpy()
        magic_range = f"[{magic_values.min():.4f}, {magic_values.max():.4f}]"
        magic_mean = magic_values.mean()
        magic_std = magic_values.std()
        
        has_attention = representations['attention_weights'] is not None
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Pooling Analysis Report - {self.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; 
                          border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 15px; padding: 15px; 
                         background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; 
                      border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; 
                           padding: 15px; border-radius: 8px; margin: 10px 0; }}
                .success {{ background: #d4edda; border: 1px solid #00b894; 
                           padding: 15px; border-radius: 8px; margin: 10px 0; }}
                h1, h2 {{ color: #2c3e50; }}
                .top-features {{ background: #f8f9ff; padding: 15px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Attention Pooling Analysis Report</h1>
                <h2>Model: {self.model_name}</h2>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Pooling Method: <strong>{self.model.pooling_type}</strong></p>
            </div>
            
            <div class="section">
                <h2>📊 Dataset Summary</h2>
                <div class="metric">
                    <strong>Samples Analyzed:</strong> {n_samples:,}
                </div>
                <div class="metric">
                    <strong>Pooled Feature Dimension:</strong> {pooled_dim}
                </div>
                <div class="metric">
                    <strong>Magic Value Range:</strong> {magic_range}
                </div>
                <div class="metric">
                    <strong>Magic Mean ± Std:</strong> {magic_mean:.4f} ± {magic_std:.4f}
                </div>
            </div>
        """
        
        if has_attention:
            html_content += f"""
            <div class="section">
                <h2>🎯 Attention Pattern Analysis</h2>
                <div class="success">
                    <strong>✅ Attention weights successfully extracted!</strong> 
                    This model uses attention-based pooling, allowing us to visualize which 
                    matrix elements the model focuses on when making predictions.
                </div>
                <p>The attention pattern analysis reveals:</p>
                <ul>
                    <li><strong>Spatial Attention:</strong> Which regions of the quantum density matrix receive the most attention</li>
                    <li><strong>Magic-Dependent Patterns:</strong> How attention differs between high and low magic states</li>
                    <li><strong>Physics Insights:</strong> Whether the model prioritizes diagonal vs off-diagonal elements</li>
                </ul>
                <img src="attention_patterns.png" alt="Attention Pattern Analysis">
            </div>
            """
        else:
            html_content += f"""
            <div class="section">
                <h2>🎯 Pooling Method Analysis</h2>
                <div class="warning">
                    <strong>ℹ️ No attention weights available.</strong> 
                    This model uses <strong>{self.model.pooling_type}</strong> pooling, 
                    which doesn't provide attention weights for visualization.
                </div>
                <p>Analysis focuses on the final pooled representations rather than attention patterns.</p>
            </div>
            """
        
        html_content += f"""
            <div class="section">
                <h2>🗺️ Feature Space Visualization</h2>
                <p>These plots show how the attention-pooled features organize quantum states in reduced dimensions:</p>
                
                <h3>2D Visualizations</h3>
                <div class="grid">
                    <div>
                        <h4>PCA (Linear Reduction)</h4>
                        <img src="pca_2d_pooled_features.png" alt="PCA 2D">
                    </div>
                    <div>
                        <h4>UMAP (Non-linear, Global Structure)</h4>
                        <img src="umap_2d_pooled_features.png" alt="UMAP 2D">
                    </div>
                </div>
                <div style="text-align: center; margin: 20px 0;">
                    <h4>t-SNE (Non-linear, Local Structure)</h4>
                    <img src="tsne_2d_pooled_features.png" alt="t-SNE 2D" style="max-width: 60%;">
                </div>
                
                <h3>3D Visualizations</h3>
                <div class="grid">
                    <div>
                        <h4>PCA 3D</h4>
                        <img src="pca_3d_pooled_features.png" alt="PCA 3D">
                    </div>
                    <div>
                        <h4>UMAP 3D</h4>
                        <img src="umap_3d_pooled_features.png" alt="UMAP 3D">
                    </div>
                </div>
                <div style="text-align: center; margin: 20px 0;">
                    <h4>t-SNE 3D</h4>
                    <img src="tsne_3d_pooled_features.png" alt="t-SNE 3D" style="max-width: 60%;">
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Feature Importance Analysis</h2>
                <p>Analysis of which pooled feature dimensions are most predictive of magic values:</p>
                
                <div class="top-features">
                    <h4>🏆 Top {len(top_features)} Most Predictive Dimensions:</h4>
                    <ul>
        """
        
        for dim, corr in top_features[:5]:
            html_content += f"<li>Dimension {dim}: correlation = {corr:.4f}</li>"
        
        html_content += f"""
                    </ul>
                </div>
                
                <img src="feature_importance.png" alt="Feature Importance Analysis">
            </div>
            
            <div class="section">
                <h2>💡 Key Insights</h2>
                <ul>
                    <li>The model compresses quantum information into <strong>{pooled_dim} dimensions</strong> using {self.model.pooling_type} pooling</li>
                    <li>Top <strong>{len(top_features)}</strong> dimensions capture the most predictive patterns for magic detection</li>
                    <li>Magic value distribution: μ={magic_mean:.4f}, σ={magic_std:.4f}</li>
        """
        
        if has_attention:
            html_content += """
                    <li><strong>Attention patterns</strong> reveal which quantum matrix elements are most important for predictions</li>
                    <li>The model shows <strong>physics-informed attention</strong> that may align with quantum mechanical principles</li>
            """
        else:
            html_content += f"""
                    <li>The <strong>{self.model.pooling_type} pooling</strong> method provides robust quantum state representation</li>
                    <li>Analysis focuses on the final pooled features rather than intermediate attention patterns</li>
            """
        
        html_content += f"""
                </ul>
            </div>
            
            <div class="section">
                <h2>📁 Generated Files</h2>
                <h4>Dimensionality Reduction Visualizations:</h4>
                <ul>
                    <li>pca_2d_pooled_features.png & pca_3d_pooled_features.png</li>
                    <li>umap_2d_pooled_features.png & umap_3d_pooled_features.png</li>
                    <li>tsne_2d_pooled_features.png & tsne_3d_pooled_features.png</li>
                </ul>
                
                <h4>Analysis Files:</h4>
                <ul>
                    <li>feature_importance.png - Most predictive feature dimensions</li>
        """
        
        if has_attention:
            html_content += "<li>attention_patterns.png - Attention weight visualizations</li>"
        
        html_content += """
                    <li>analysis_report.html - This comprehensive report</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Generated HTML report: {report_path}")


def load_model_with_config(model_path: str, model_config: Dict) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a quantum magic prediction model with the given configuration.
    
    Args:
        model_path: Path to the saved model weights
        model_config: Dictionary containing model configuration
        
    Returns:
        tuple: (loaded_model, device)
    """
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CLS_Usage", model_config.get('use_cls_token', True))
    # Create model with configuration
    model = CompleteQuantumMagicPredictor(
        matrix_dim=model_config['matrix_dim'],
        d_model=model_config['d_model'],
        pooling_type=model_config['pooling_type'],
        mlp_type=model_config['mlp_type'],
        use_physics_mask=model_config['use_physics_mask'],
        mask_threshold=model_config.get('mask_threshold', 1),
        nhead=model_config.get('nhead', 8),
        use_cls_token=model_config.get('use_cls_token', True)
    )
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Loading with strict=False due to: {e}")
        state_dict = torch.load(model_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        return
    
    return model, device


def analyze_attention_pooling_model(model_path: str, data_folder: str, labels_path: str, 
                                  model_config: Dict, output_dir: Optional[str] = None):
    """
    Complete analysis pipeline for attention-based pooling models.
    
    Args:
        model_path: Path to trained model weights
        data_folder: Folder containing quantum density matrix data
        labels_path: Path to magic value labels
        model_config: Model configuration dictionary
        output_dir: Optional custom output directory
        
    Returns:
        tuple: (analyzer, representations, top_features) for further analysis
    """
    from Model_Archi.quantum_dataset import create_dataloaders
    
    # Extract model name from path
    model_name = Path(model_path).stem
    
    print(f"🚀 Starting attention pooling analysis for: {model_name}")
    print(f"📁 Model: {model_path}")
    print(f"📁 Data: {data_folder}")
    print(f"📁 Labels: {labels_path}")
    
    # Load model
    print("🔧 Loading model...")
    model, device = load_model_with_config(model_path, model_config)
    
    # Create dataloader
    print("📊 Creating data loaders...")
    _, _, test_loader = create_dataloaders(
        data_folder=data_folder,
        labels_path=labels_path,
        batch_size=256,
        num_workers=4,
        validate=True
    )
    
    # Create analyzer
    analyzer = AttentionPoolingVisualizationTool(
        model=model,
        device=device,
        model_name=model_name
    )
    
    # Generate comprehensive analysis
    representations, top_features = analyzer.generate_comprehensive_report(
        test_loader, output_dir=output_dir
    )
    
    print(f"🎉 Analysis complete!")
    
    return analyzer, representations, top_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze quantum magic prediction models with attention-based pooling"
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the trained model weights (.pth file)")
    parser.add_argument('--config_from_json', type=str, required=True,
                       help="Path to model configuration JSON file")
    
    # Data arguments
    parser.add_argument('--data_folder', type=str, default="Rawdata/DecodedTokens",
                       help="Folder containing quantum density matrix data")
    parser.add_argument('--labels_path', type=str, 
                       default="Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy",
                       help="Path to magic value labels")
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Custom output directory (default: visualizations/{model_name}/)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_from_json, 'r') as f:
        model_config = json.load(f)
    
    print("🔧 Model Configuration:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")
    
    # Run analysis
    try:
        analyzer, representations, top_features = analyze_attention_pooling_model(
            model_path=args.model_path,
            data_folder=args.data_folder,
            labels_path=args.labels_path,
            model_config=model_config,
            output_dir=args.output_dir
        )
        
        print("✅ Analysis completed successfully!")
        print(f"🎯 Top 3 predictive dimensions: {[(d, f'{c:.3f}') for d, c in top_features[:3]]}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()