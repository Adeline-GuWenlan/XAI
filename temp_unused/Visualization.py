import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from torch.utils.data import DataLoader
import pandas as pd
# NEW: for 3D plots (older Matplotlib needs this import side-effect)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import argparse
import os
import json
from pathlib import Path
# NEW: robust perplexity for t-SNE
def _auto_tsne_perplexity(n_samples: int) -> int:
    """
    Choose a safe perplexity given n_samples.
    Ensures 2 <= perplexity < n_samples, and roughly keeps 3*perplexity <= n_samples.
    """
    if n_samples <= 3:
        return 2
    # aim for ~ n/3, clamp to [5, 30], then ensure < n
    p = min(30, max(5, (n_samples - 1) // 3))
    p = max(2, min(p, n_samples - 1))
    return int(p)

class CLSVisualizationTool:
    """Tool to visualize what the CLS token learns - with unique output naming"""
    
    def __init__(self, model, device='cpu', output_prefix="analysis"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.output_prefix = output_prefix
    def extract_representations(self, dataloader, max_samples=150000):

        import torch
        import numpy as np

        # ---------- helpers ----------
        def _move_to(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device, non_blocking=True)
            if isinstance(obj, dict):
                return {k: _move_to(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                seq = [_move_to(v, device) for v in obj]
                return tuple(seq) if isinstance(obj, tuple) else seq
            return obj

        def _to_tensor_1d(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().flatten().cpu()
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).flatten().cpu()
            # fall back: try to tensor-ize list/py types
            try:
                return torch.as_tensor(x).flatten().cpu()
            except Exception:
                return None

        def _parse_batch(batch):
            """
            Return (model_inputs, labels) without assuming exact structure.
            - tuple/list: inputs = all but last, labels = last (if len>=2); if len==1 -> no labels
            - dict: labels = any of common label keys (if present); inputs = the rest
            - other: inputs = batch, labels = None
            """
            if isinstance(batch, (list, tuple)):
                if len(batch) == 0:
                    raise RuntimeError("Empty batch.")
                if len(batch) == 1:
                    return batch[0], None
                # inputs = all but last, labels = last
                model_inputs = batch[:-1]
                labels = batch[-1]
                if len(model_inputs) == 1:
                    model_inputs = model_inputs[0]
                return model_inputs, labels

            if isinstance(batch, dict):
                label_keys = ["labels", "label", "y", "target", "targets"]
                labels = None
                for k in label_keys:
                    if k in batch:
                        labels = batch[k]
                        break
                # inputs = everything except the chosen label key (if any)
                if labels is None:
                    model_inputs = batch
                else:
                    model_inputs = {k: v for k, v in batch.items() if k != k}  # placeholder, replaced below
                    # fix: we excluded nothing above (can't know which key we used). Build inputs cleanly:
                    model_inputs = {k: v for k, v in batch.items() if k not in label_keys}
                # If dict has typical transformer keys, pass only those
                if any(k in model_inputs for k in ("input_ids", "attention_mask", "token_type_ids")):
                    model_inputs = {k: model_inputs[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in model_inputs}
                return model_inputs, labels

            # anything else
            return batch, None

        def _model_forward(model, model_inputs):
            # Special handling for our quantum model to extract CLS representations
            if hasattr(model, 'extract_cls_representations'):
                return model.extract_cls_representations(*model_inputs)
            elif isinstance(model_inputs, dict):
                return model(**model_inputs)
            elif isinstance(model_inputs, (list, tuple)):
                return model(*model_inputs)
            else:
                return model(model_inputs)

        def _pick_first_tensor(seq, dim=None):
            """Pick the first tensor (optionally matching a required ndim) from a list/tuple."""
            for item in seq:
                if torch.is_tensor(item) and (dim is None or item.dim() == dim):
                    return item
            return None

        def _parse_outputs(out):
            """
            Try hard to get (rep_batch, cls_tokens) from diverse outputs.
            - rep_batch: [B, L, D] preferred. If only [B, D], we'll unsqueeze to [B,1,D].
            - cls_tokens: [B, D] if available; else derive from rep_batch[:,0,:] if L>=1.
            """
            rep_batch = None
            cls_tokens = None

            if isinstance(out, dict):
                # token-level candidates
                for k in ("token_embeddings", "tokens", "last_hidden_state", "sequence_output"):
                    if k in out and torch.is_tensor(out[k]):
                        rep_batch = out[k]
                        break
                # hidden_states as list/tuple -> use last
                if rep_batch is None and "hidden_states" in out and isinstance(out["hidden_states"], (list, tuple)) and out["hidden_states"]:
                    last = out["hidden_states"][-1]
                    if torch.is_tensor(last):
                        rep_batch = last

                # cls-level candidates
                for k in ("cls", "cls_tokens", "pooled", "pooler_output"):
                    if k in out and torch.is_tensor(out[k]):
                        cls_tokens = out[k]
                        break

            elif isinstance(out, (list, tuple)):
                # prefer a 3D tensor as token embeddings
                rep_batch = _pick_first_tensor(out, dim=3)
                # next, look for a 2D tensor as CLS
                cls_tokens = _pick_first_tensor(out, dim=2)
                # also search any dicts inside
                for item in out:
                    if isinstance(item, dict):
                        r2, c2 = _parse_outputs(item)
                        rep_batch = rep_batch if rep_batch is not None else r2
                        cls_tokens = cls_tokens if cls_tokens is not None else c2

            elif torch.is_tensor(out):
                if out.dim() == 3:
                    rep_batch = out
                elif out.dim() == 2:
                    cls_tokens = out

            # normalize shapes
            if rep_batch is not None and rep_batch.dim() == 2:
                rep_batch = rep_batch.unsqueeze(1)  # [B, D] -> [B,1,D]
            if cls_tokens is None and rep_batch is not None and rep_batch.dim() == 3 and rep_batch.size(1) >= 1:
                # fallback: take the first token as "CLS-like"
                cls_tokens = rep_batch[:, 0, :]

            return rep_batch, cls_tokens

        # ---------- main ----------
        device = next(self.model.parameters()).device
        buf = {"representations": [], "cls_tokens": [], "labels": []}

        self.model.eval()
        seen = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch_count += 1
                
                model_inputs, labels = _parse_batch(batch)
                model_inputs = _move_to(model_inputs, device)
                out = _model_forward(self.model, model_inputs)
                rep_batch, cls_batch = _parse_outputs(out)

                if rep_batch is None and cls_batch is None:
                    raise RuntimeError(
                        "Model outputs did not contain token-level or CLS embeddings. "
                        "Return a dict with keys like 'last_hidden_state' or 'cls' from your model forward."
                    )

                # ensure both exist & normalized
                if rep_batch is None and cls_batch is not None:
                    rep_batch = cls_batch.unsqueeze(1)  # [B,1,D]
                if cls_batch is None and rep_batch is not None:
                    cls_batch = rep_batch[:, 0, :]      # [B,D] from first token

                # move to CPU for accumulation
                rep_batch = rep_batch.detach().cpu()
                cls_batch = cls_batch.detach().cpu()
                lab_batch = _to_tensor_1d(labels)

                buf["representations"].append(rep_batch)  # [B,L,D]
                buf["cls_tokens"].append(cls_batch)        # [B,D]
                if lab_batch is not None:
                    buf["labels"].append(lab_batch)        # [B]

                seen += rep_batch.size(0)
                if seen >= max_samples:
                    break

        # concat
        if buf["representations"]:
            buf["representations"] = torch.cat(buf["representations"], dim=0)  # [N,L,D]
        else:
            buf["representations"] = None

        if buf["cls_tokens"]:
            buf["cls_tokens"] = torch.cat(buf["cls_tokens"], dim=0)            # [N,D]
        else:
            buf["cls_tokens"] = None

        if buf["labels"]:
            # if label batches have mixed sizes/dtypes, torch.cat will still work after _to_tensor_1d
            buf["labels"] = torch.cat(buf["labels"], dim=0)                     # [N]
        else:
            buf["labels"] = None

        # sanity checks (fail fast with clear messages)
        if buf["representations"] is not None and buf["representations"].dim() != 3:
            raise RuntimeError(f"Expected [N,L,D] for 'representations', got {tuple(buf['representations'].shape)}")
        if buf["cls_tokens"] is not None and buf["cls_tokens"].dim() != 2:
            raise RuntimeError(f"Expected [N,D] for 'cls_tokens', got {tuple(buf['cls_tokens'].shape)}")

        return buf


    def visualize_cls_space(self, representations, method="umap", dim=2, out_path=None):
        """
        Visualize CLS token space with UMAP / t-SNE / PCA in 2D or 3D.

        Args:
            representations: dict with 'cls_tokens' and 'labels' keys
            method:     "umap" | "tsne" | "pca"
            dim:        2 or 3 (number of components & plot dimensionality)
            out_path:   optional path to save the figure
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        try:
            from umap import UMAP
        except Exception:
            UMAP = None

        assert dim in (2, 3), "dim must be 2 or 3"
        
        # Extract CLS tokens and labels from representations dict
        cls_tokens = representations['cls_tokens']
        labels = representations['labels']
        # Convert torch -> numpy if needed
        if hasattr(cls_tokens, "detach"):
            cls_tokens = cls_tokens.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        n = int(len(cls_tokens))
        n_features = cls_tokens.shape[1] if len(cls_tokens.shape) > 1 else 1
        
        # Validate sample size for dimensionality reduction
        if n < dim:
            print(f"⚠️  Warning: Only {n} samples available, cannot create {dim}D visualization")
            print(f"⚠️  Skipping {method} {dim}D visualization")
            return
            
        if n_features < dim:
            print(f"⚠️  Warning: Only {n_features} features available, cannot create {dim}D visualization") 
            print(f"⚠️  Skipping {method} {dim}D visualization")
            return
            
        # Additional check for minimum samples needed for each method
        min_samples_needed = max(dim + 1, 10)  # At least dim+1 samples, preferably 10+
        if method.lower() == 'tsne':
            min_samples_needed = max(dim + 1, 30)  # t-SNE needs more samples
            
        if n < min_samples_needed:
            print(f"⚠️  Warning: {method} needs at least {min_samples_needed} samples, got {n}")
            print(f"⚠️  Skipping {method} {dim}D visualization")
            return

        # -------- reducers (now parameterized by dim) --------
        method = method.lower()
        if method == "pca":
            reducer = PCA(n_components=dim, random_state=42)
            emb = reducer.fit_transform(cls_tokens)

        elif method == "tsne":
            perplexity = _auto_tsne_perplexity(n)
            reducer = TSNE(
                n_components=dim,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            )
            emb = reducer.fit_transform(cls_tokens)

        elif method == "umap":
            if UMAP is None:
                raise RuntimeError("umap-learn is not installed. `pip install umap-learn`")
            reducer = UMAP(
                n_components=dim,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            )
            emb = reducer.fit_transform(cls_tokens)

        else:
            raise ValueError(f"Unknown method: {method}")

        # -------- plotting (2D vs 3D) --------
        title = f"{method.upper()} ({dim}D)"
        if dim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", alpha=0.6, s=20)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(title)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Label")
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=labels, cmap="viridis", alpha=0.6, s=12, depthshade=True)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)

        if out_path:
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def analyze_cls_components(self, representations, top_k=10):
        """Analyze which components of CLS token are most important"""
        cls_tokens = representations['cls_tokens']
        labels = representations['labels']
        
        # Ensure labels is 1D
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        # Ensure same number of samples
        min_samples = min(cls_tokens.shape[0], labels.shape[0])
        cls_tokens = cls_tokens[:min_samples]
        labels = labels[:min_samples]
        
        # Compute correlations
        correlations = []
        for i in range(cls_tokens.shape[1]):
            try:
                corr = np.corrcoef(cls_tokens[:, i], labels)[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, abs(corr)))
            except Exception as e:
                print(f"Warning: Could not compute correlation for dimension {i}: {e}")
                continue
        
        if len(correlations) == 0:
            print("Warning: No valid correlations found!")
            return [], plt.figure()
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Plot top correlations
        plt.figure(figsize=(12, 6))
        dims, corr_vals = zip(*correlations[:top_k])
        plt.bar(range(len(dims)), corr_vals)
        plt.xlabel('CLS Token Dimension')
        plt.ylabel('Absolute Correlation with Magic')
        plt.title(f'Top {top_k} CLS Dimensions by Correlation with Magic Value')
        plt.xticks(range(len(dims)), [f'Dim {d}' for d in dims], rotation=45)
        plt.tight_layout()
        
        return correlations[:top_k], plt.gcf()
    
    def visualize_attention_patterns(self, dataloader, sample_idx=0):
        """Visualize attention patterns"""
        with torch.no_grad():
            for i, (rho_real, rho_imag, labels) in enumerate(dataloader):
                if i < sample_idx:
                    continue
                    
                rho_real = rho_real.to(self.device)
                rho_imag = rho_imag.to(self.device)
                
                sample_real = rho_real[0].cpu().numpy()
                sample_imag = rho_imag[0].cpu().numpy()
                sample_label = labels[0].item() if hasattr(labels[0], 'item') else labels[0]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Real part
                real_max = max(abs(sample_real.min()), abs(sample_real.max()))
                im1 = axes[0,0].imshow(sample_real, cmap='RdBu', vmin=-real_max, vmax=real_max)
                axes[0,0].set_title('Real Part of Density Matrix')
                plt.colorbar(im1, ax=axes[0,0])
                
                # Imaginary part
                imag_max = max(abs(sample_imag.min()), abs(sample_imag.max()))
                im2 = axes[0,1].imshow(sample_imag, cmap='RdBu', vmin=-imag_max, vmax=imag_max)
                axes[0,1].set_title('Imaginary Part of Density Matrix')
                plt.colorbar(im2, ax=axes[0,1])
                
                # Combined magnitude
                magnitude = np.sqrt(sample_real**2 + sample_imag**2)
                im3 = axes[1,0].imshow(magnitude, cmap='viridis')
                axes[1,0].set_title('Magnitude |ρ|')
                plt.colorbar(im3, ax=axes[1,0])
                
                # Add text info
                axes[1,1].text(0.1, 0.8, f'Magic Value: {sample_label:.4f}', 
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].text(0.1, 0.6, f'Matrix Dimension: {sample_real.shape[0]}x{sample_real.shape[1]}', 
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].text(0.1, 0.4, f'Trace: {np.trace(sample_real + 1j*sample_imag):.4f}', 
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].axis('off')
                
                plt.suptitle(f'Sample {sample_idx}: Input Quantum State')
                plt.tight_layout()
                break
                
        return plt.gcf()
    
    def compare_embeddings_vs_cls(self, representations):
        """Compare initial matrix embeddings vs final CLS representations"""
        # Check if matrix_embeddings exist, otherwise use first layer representations
        if 'matrix_embeddings' in representations:
            matrix_emb = representations['matrix_embeddings']
        else:
            # Fallback: use first token from representations as matrix embeddings
            matrix_emb = representations['representations'][:, 0, :]
        
        cls_tokens = representations['cls_tokens']
        labels = representations['labels']
        
        # Ensure labels is 1D
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        # Ensure same number of samples
        min_samples = min(matrix_emb.shape[0], cls_tokens.shape[0], labels.shape[0])
        matrix_emb = matrix_emb[:min_samples]
        cls_tokens = cls_tokens[:min_samples]
        labels = labels[:min_samples]
        
        # Compute correlations
        matrix_label_corr = []
        cls_label_corr = []
        
        for i in range(min(matrix_emb.shape[1], cls_tokens.shape[1])):
            try:
                corr_m = np.corrcoef(matrix_emb[:, i], labels)[0, 1]
                if not np.isnan(corr_m):
                    matrix_label_corr.append(abs(corr_m))
                else:
                    matrix_label_corr.append(0.0)
                
                corr_c = np.corrcoef(cls_tokens[:, i], labels)[0, 1]
                if not np.isnan(corr_c):
                    cls_label_corr.append(abs(corr_c))
                else:
                    cls_label_corr.append(0.0)
            except Exception as e:
                print(f"Warning: Could not compute correlation for dimension {i}: {e}")
                matrix_label_corr.append(0.0)
                cls_label_corr.append(0.0)
                continue
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        x = range(len(matrix_label_corr))
        plt.plot(x, matrix_label_corr, 'b-', label='Initial Matrix Embeddings', alpha=0.7)
        plt.plot(x, cls_label_corr, 'r-', label='Final CLS Token', alpha=0.7)
        plt.xlabel('Dimension Index')
        plt.ylabel('Absolute Correlation with Magic')
        plt.title('Information Processing: Matrix Embeddings → CLS Token')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_report(self, dataloader, output_dir=None):
        """Generate comprehensive analysis report with unique naming"""
        if output_dir is None:
            output_dir = Path(".")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Extracting representations...")
        representations = self.extract_representations(dataloader, max_samples=150000)
        
        print("📊 Generating visualizations...")
        
        # Create all visualizations with unique names
        figs = {}
        
        # UMAP visualizations
        print("📊 Generating UMAP 2D visualization...")
        self.visualize_cls_space(representations, method='umap', dim=2, 
                                out_path=output_dir / f"{self.output_prefix}_umap_2d.png")
        print("📊 Generating UMAP 3D visualization...")
        self.visualize_cls_space(representations, method='umap', dim=3, 
                                out_path=output_dir / f"{self.output_prefix}_umap_3d.png")
        
        # t-SNE visualizations
        print("📊 Generating t-SNE 2D visualization...")
        self.visualize_cls_space(representations, method='tsne', dim=2, 
                                out_path=output_dir / f"{self.output_prefix}_tsne_2d.png")
        print("📊 Generating t-SNE 3D visualization...")
        self.visualize_cls_space(representations, method='tsne', dim=3, 
                                out_path=output_dir / f"{self.output_prefix}_tsne_3d.png")
        
        # PCA visualizations
        print("📊 Generating PCA 2D visualization...")
        self.visualize_cls_space(representations, method='pca', dim=2, 
                                out_path=output_dir / f"{self.output_prefix}_pca_2d.png")
        print("📊 Generating PCA 3D visualization...")
        self.visualize_cls_space(representations, method='pca', dim=3, 
                                out_path=output_dir / f"{self.output_prefix}_pca_3d.png")
        
        # Generate remaining analysis figures
        figs = {}
        top_dims, figs['components'] = self.analyze_cls_components(representations)
        figs['attention'] = self.visualize_attention_patterns(dataloader, sample_idx=0)
        figs['comparison'] = self.compare_embeddings_vs_cls(representations)
        
        # Save remaining figures with unique names
        for name, fig in figs.items():
            filename = output_dir / f"{self.output_prefix}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {filename}")
            plt.close(fig)
        
        # Generate HTML report with unique name
        report_path = output_dir / f"{self.output_prefix}_report.html"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CLS Token Analysis Report - {self.output_prefix}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background: #f0f0f0; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .header {{ background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CLS Token Analysis Report</h1>
                <h2>Experiment: {self.output_prefix}</h2>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Dataset Summary</h2>
                <div class="metric">
                    <strong>Samples Analyzed:</strong> {len(representations['labels'])}
                </div>
                <div class="metric">
                    <strong>CLS Dimension:</strong> {representations['cls_tokens'].shape[1]}
                </div>
                <div class="metric">
                    <strong>Magic Range:</strong> [{representations['labels'].min():.4f}, {representations['labels'].max():.4f}]
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 CLS Token Space Visualization</h2>
                <p>These plots show how the CLS token organizes quantum states:</p>
                <h3>2D Visualizations</h3>
                <img src="{self.output_prefix}_umap_2d.png" alt="UMAP 2D visualization">
                <img src="{self.output_prefix}_tsne_2d.png" alt="t-SNE 2D visualization">
                <img src="{self.output_prefix}_pca_2d.png" alt="PCA 2D visualization">
                
                <h3>3D Visualizations</h3>
                <img src="{self.output_prefix}_umap_3d.png" alt="UMAP 3D visualization">
                <img src="{self.output_prefix}_tsne_3d.png" alt="t-SNE 3D visualization">
                <img src="{self.output_prefix}_pca_3d.png" alt="PCA 3D visualization">
            </div>
            
            <div class="section">
                <h2>📈 Most Important CLS Dimensions</h2>
                <p>Dimensions most correlated with magic monotone values:</p>
                <ul>
        """
        
        for dim, corr in top_dims[:5]:
            html_content += f"<li>Dimension {dim}: correlation = {corr:.4f}</li>"
        
        html_content += f"""
                </ul>
                <img src="{self.output_prefix}_components.png" alt="Component analysis">
            </div>
            
            <div class="section">
                <h2>🔍 Sample Input Analysis</h2>
                <p>Example of input quantum state:</p>
                <img src="{self.output_prefix}_attention.png" alt="Sample input">
            </div>
            
            <div class="section">
                <h2>🔄 Information Flow Analysis</h2>
                <p>How information flows from embeddings to CLS:</p>
                <img src="{self.output_prefix}_comparison.png" alt="Information flow">
            </div>
            
            <div class="section">
                <h2>💡 Key Insights</h2>
                <ul>
                    <li>The CLS token compresses quantum state information into {representations['cls_tokens'].shape[1]} dimensions</li>
                    <li>Top {len(top_dims)} most predictive dimensions capture the majority of magic detection capability</li>
                    <li>The transformer appears to {"successfully" if top_dims[0][1] > 0.3 else "struggle to"} extract meaningful patterns from quantum matrices</li>
                    <li>Magic value distribution: μ={representations['labels'].mean():.4f}, σ={representations['labels'].std():.4f}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📁 Output Files</h2>
                <ul>
                    <li><strong>{self.output_prefix}_report.html</strong> - This report</li>
                    <li><strong>{self.output_prefix}_umap_2d.png</strong> - UMAP 2D visualization</li>
                    <li><strong>{self.output_prefix}_umap_3d.png</strong> - UMAP 3D visualization</li>
                    <li><strong>{self.output_prefix}_tsne_2d.png</strong> - t-SNE 2D visualization</li>
                    <li><strong>{self.output_prefix}_tsne_3d.png</strong> - t-SNE 3D visualization</li>
                    <li><strong>{self.output_prefix}_pca_2d.png</strong> - PCA 2D visualization</li>
                    <li><strong>{self.output_prefix}_pca_3d.png</strong> - PCA 3D visualization</li>
                    <li><strong>{self.output_prefix}_components.png</strong> - Component analysis</li>
                    <li><strong>{self.output_prefix}_attention.png</strong> - Sample input visualization</li>
                    <li><strong>{self.output_prefix}_comparison.png</strong> - Information flow analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Analysis complete! Report saved to: {report_path}")
        print(f"📊 Generated {len(figs)} visualization files")
        print(f"🎯 Top correlating dimensions: {[f'Dim{d}({c:.3f})' for d,c in top_dims[:3]]}")
        
        return representations, top_dims


def load_model_with_flexible_config(model_path, model_config):
    """Load model with flexible configuration handling - UPDATED"""
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ✅ FIXED: Use correct parameter names matching updated_training_model.py
    model = CompleteQuantumMagicPredictor(
        matrix_dim=model_config['matrix_dim'],
        d_model=model_config['d_model'],
        pooling_type=model_config['pooling_type'],
        mlp_type=model_config['mlp_type'],
        use_physics_mask=model_config['use_physics_mask'],  # ✅ FIXED: Correct parameter name
        mask_threshold=model_config.get('mask_threshold', 1),  # ✅ NEW: Added with default
        nhead=model_config.get('nhead', 8),  # ✅ NEW: Added with default
        use_cls_token=model_config.get('use_cls_token', True)  # ✅ NEW: Added with default
    )
    
    # Load with flexible handling
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully (strict mode)")
    except RuntimeError as e:
        if 'physics_mask' in str(e):
            print("⚠️ Physics mask mismatch detected, loading with strict=False...")
            state_dict = torch.load(model_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            # Manually initialize missing physics_mask if needed
            for name, module in model.named_modules():
                if hasattr(module, 'physics_mask') and module.physics_mask is None:
                    if hasattr(module, '_create_physics_mask'):
                        module.register_buffer('physics_mask', module._create_physics_mask())
                        print(f"✅ Recreated physics_mask for {name}")
            
            print("✅ Model loaded successfully (flexible mode)")
        else:
            raise e
    
    return model, device


def analyze_trained_model(model_path, data_folder, labels_path, model_config, output_prefix="analysis", output_dir=None):
    """Complete analysis of a trained model with unique outputs"""
    from Model_Archi.quantum_dataset import create_dataloaders
    
    print(f"🔧 Model Configuration:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")
    
    # Load model
    model, device = load_model_with_flexible_config(model_path, model_config)
    
    # Add method to extract CLS representations instead of final predictions
    def extract_cls_representations(self, rho_real, rho_imag):
        """Extract CLS token representations before final MLP"""
        # Embedding: [B, D, D] → [B, D², d_model] (matrix tokens only)
        matrix_tokens = self.embedding(rho_real, rho_imag)
        B = matrix_tokens.shape[0]
        
        # Add CLS token if using CLS pooling
        if self.pooling_type == "cls" and hasattr(self, 'cls_token'):
            # Expand CLS token for batch
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            tokens = torch.cat([cls_tokens, matrix_tokens], dim=1)  # [B, D²+1, d_model]
        else:
            tokens = matrix_tokens
        
        # Transformer processing
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        
        # Extract features before final MLP
        if self.pooling_type == "cls":
            # Use CLS token (position 0) - these are our CLS representations
            cls_representations = tokens[:, 0, :]  # [B, d_model]
            
            # Return both full token sequences and CLS tokens
            return {
                'last_hidden_state': tokens,      # [B, seq_len, d_model] 
                'cls': cls_representations,       # [B, d_model]
                'pooler_output': cls_representations  # Alias for compatibility
            }
        else:
            # Use physics-aware pooling
            global_features = self.physics_pooling(tokens)
            return {
                'last_hidden_state': tokens,
                'cls': global_features,
                'pooler_output': global_features
            }
    
    # Bind method to model instance
    import types
    model.extract_cls_representations = types.MethodType(extract_cls_representations, model)
    
    # Create dataloader
    print(f"🔍 Creating dataloaders with:")
    print(f"   data_folder: {data_folder}")
    print(f"   labels_path: {labels_path}")
    
    test_loader, _, _ = create_dataloaders(
        data_folder=data_folder,
        labels_path=labels_path,
        batch_size=256,
        num_workers=8
    )
    
    # Dataloader created successfully
    # Run analysis with unique naming
    analyzer = CLSVisualizationTool(model, device, output_prefix=output_prefix)
    representations, top_dims = analyzer.generate_report(test_loader, output_dir=output_dir)
    
    return analyzer, representations, top_dims


def load_config_from_json(json_path):
    """Load configuration from JSON file - UPDATED"""
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trained quantum magic prediction models with isolated outputs")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                        help="Path to the trained model weights (.pth file)")
    parser.add_argument('--data_folder', type=str, default="Rawdata/DecodedTokens", 
                        help="Folder containing input quantum matrices")
    parser.add_argument('--labels_path', type=str, default="Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy", 
                        help="Path to the labels file")
    
    # Configuration source
    parser.add_argument('--config_from_json', type=str, required=True,
                       help="Load config from JSON file")
    
    # Output control
    parser.add_argument('--output_prefix', type=str, default=None,
                       help="Prefix for output files (auto-generated if not specified)")
    parser.add_argument('--output_dir', type=str, default="visualizations",
                       help="Directory for visualization outputs")
    
    # Manual configuration (fallback) - ✅ UPDATED parameter names
    parser.add_argument('--n_qubits', type=int, default=2, 
                        help="Number of qubits")
    parser.add_argument('--d_model', type=int, default=128, 
                        help="Model dimension")
    parser.add_argument('--use_physics_mask',  # ✅ FIXED: Correct parameter name
                        type=lambda x: x.lower() == 'true', 
                        default=False, 
                        help="Use physics mask (True/False)")
    parser.add_argument('--mask_threshold', type=int, default=1,  # ✅ NEW: Added parameter
                        help="Physics mask threshold")
    parser.add_argument('--nhead', type=int, default=8,  # ✅ NEW: Added parameter
                        help="Number of attention heads")
    parser.add_argument('--use_cls_token',  # ✅ NEW: Added parameter
                        type=lambda x: x.lower() == 'true', 
                        default=True, 
                        help="Use CLS token (True/False)")
    parser.add_argument('--pooling_type', type=str, default="cls",
                        choices=["cls", "mean", "attention", "structured"],
                        help="Pooling type")
    parser.add_argument('--mlp_type', type=str, default="standard", 
                        choices=["standard", "physics_aware", "attention_enhanced", 
                               "mixture_of_experts", "asymmetric_ensemble"],
                        help="MLP type")
    
    args = parser.parse_args()
    
    # Generate unique output prefix if not provided
    if args.output_prefix is None:
        model_name = Path(args.model_path).stem
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_prefix = f"{model_name}_viz_{timestamp}"
    
    # Load configuration
    model_config = {}
    
    model_config = load_config_from_json(args.config_from_json)

    print(f"🚀 Starting visualization analysis...")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_folder}")
    print(f"📁 Labels: {args.labels_path}")
    print(f"📂 Output: {args.output_dir}/{args.output_prefix}_*")
    
    try:
        analyzer, representations, top_dims = analyze_trained_model(
            model_path=args.model_path,
            data_folder=args.data_folder,
            labels_path=args.labels_path,
            model_config=model_config,
            output_prefix=args.output_prefix,
            output_dir=args.output_dir
        )
        
        print("✅ Analysis complete! Generated files:")
        print(f"   📊 {args.output_dir}/{args.output_prefix}_report.html - Main report")
        print(f"   📈 {args.output_dir}/{args.output_prefix}_*.png - Individual plots")
        print(f"   🎯 Top correlating dimensions: {[f'Dim{d}({c:.3f})' for d,c in top_dims[:3]]}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()