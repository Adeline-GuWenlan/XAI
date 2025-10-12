# layerwise_probe_exact.py
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List

class LayerwiseCLSProbeExact:
    """
    Fixed hooks for the provided architecture:
    - Encoder layers (4): capture CLS after each encoder layer
    - MLP head (attention_enhanced):
        * input_norm pre/post
        * feature_extractor[0] (Linear 256x128) pre/post
        * feature_extractor[2] (LayerNorm 256)  pre/post
        * attention (nn.MultiheadAttention)     post (attn_output)
        * post_attention[0] LN(256)             pre/post
        * post_attention[1] Linear(128x256)     pre/post
        * post_attention[3] LN(128)             pre/post
        * post_attention[5] Linear(64x128)      pre/post
        * post_attention[7] Linear(32x64)       pre/post
        * post_attention[9] Linear(1x32)        post (final scalar logits-like)
    """

    # Base encoder keys - always present
    BASE_ENCODER_KEYS: List[str] = [
        "enc_00_post", "enc_01_post", "enc_02_post", "enc_03_post"
    ]
    
    # Architecture-specific MLP keys
    ATTENTION_ENHANCED_KEYS: List[str] = [
        "mlp_in_pre", "mlp_in_post",
        "mlp_fe0_pre", "mlp_fe0_post", "mlp_fe2_pre", "mlp_fe2_post",
        "mlp_attn_post",
        "mlp_post_00_ln_pre", "mlp_post_00_ln_post",
        "mlp_post_01_linear_pre", "mlp_post_01_linear_post",
        "mlp_post_03_ln_pre", "mlp_post_03_ln_post",
        "mlp_post_05_linear_pre", "mlp_post_05_linear_post",
        "mlp_post_07_linear_pre", "mlp_post_07_linear_post",
        "mlp_post_09_linear_post"
    ]
    
    MOE_KEYS: List[str] = [
        "mlp_in_pre", "mlp_in_post",
        "mlp_gating_pre", "mlp_gating_post",
        "mlp_expert_00_pre", "mlp_expert_00_post",
        "mlp_expert_01_pre", "mlp_expert_01_post", 
        "mlp_expert_02_pre", "mlp_expert_02_post"
    ]
    
    GENERIC_KEYS: List[str] = [
        "mlp_in_pre", "mlp_in_post",
        "mlp_generic_pre", "mlp_generic_post"
    ]

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}
        self._hooks = []
        
        # Move model to device and set evaluation mode
        self.model.to(device).eval()
        print(f'Model moved to device: {next(model.parameters()).device}')
        
        # Optimize for inference
        torch.set_grad_enabled(False)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    @staticmethod
    def _to_vec(x: torch.Tensor) -> torch.Tensor:
        """
        Make a (B, D) vector:
        - If (B, L, D), take CLS at position 0
        - If (L, B, D) (MHA with batch_first=False), squeeze L if equals 1
        - If (B, D), keep
        - If (B, 1, D) or (1, B, D), squeeze the length-1 dim
        """
        if x.dim() == 3:
            # try batch_first (B, L, D)
            if x.shape[1] >= 1 and x.shape[0] >= 1:
                # prefer CLS (pos 0) if looks like sequence
                return x[:, 0, :]
            # try (L, B, D) with L==1
            if x.shape[0] == 1:
                return x[0, :, :]
            if x.shape[1] == 1:
                return x[:, 0, :]
        return x  # assume (B, D) already

    def register_hooks(self):
        """Register hooks for different model architectures dynamically."""
        self.remove_hooks()
        self.cache = {}

        pooling_type = getattr(self.model, "pooling_type", "cls")
        assert pooling_type == "cls", "This exact probe assumes CLS pooling is used."

        # ----- Encoder: capture CLS after each encoder layer -----
        for i, layer in enumerate(self.model.encoder_layers):
            key = f"enc_{i:02d}_post"
            def _mk_enc_hook(k):
                def _hook(module, inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    cls_vec = self._to_vec(out)
                    self.cache[k] = self._to_np(cls_vec)
                return _hook
            self._hooks.append(layer.register_forward_hook(_mk_enc_hook(key)))

        # ----- MLP head: detect and handle different architectures -----
        mlp_root = self.model.mlp_head                       # QuantumMagicMLPv2(...)
        mlp_type = getattr(mlp_root, 'mlp_type', 'unknown')
        
        # Always hook input_norm if it exists
        if hasattr(mlp_root, 'input_norm'):
            input_norm = mlp_root.input_norm
            self._register_module_hooks(input_norm, "mlp_in")
        
        # Handle different MLP architectures
        if mlp_type == "mixture_of_experts":
            self._register_moe_hooks(mlp_root)
        elif mlp_type == "attention_enhanced":
            self._register_attention_enhanced_hooks(mlp_root)
        else:
            # For other types, register hooks for the main mlp_head
            self._register_generic_mlp_hooks(mlp_root)

    def _register_module_hooks(self, module, base_name):
        """Helper to register pre/post hooks for any module."""
        def _pre_hook(mod, inputs):
            x = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else inputs
            if isinstance(x, torch.Tensor):
                self.cache[f"{base_name}_pre"] = self._to_np(x)
        
        def _post_hook(mod, inputs, output):
            y = output[0] if isinstance(output, tuple) else output
            if isinstance(y, torch.Tensor):
                self.cache[f"{base_name}_post"] = self._to_np(y)
        
        self._hooks.append(module.register_forward_pre_hook(_pre_hook))
        self._hooks.append(module.register_forward_hook(_post_hook))

    def _register_moe_hooks(self, mlp_root):
        """Register hooks for mixture_of_experts MLP architecture."""
        mh = mlp_root.mlp_head  # ModuleDict with 'gating_network' and 'experts'
        
        # Hook gating network
        if 'gating_network' in mh:
            self._register_module_hooks(mh['gating_network'], "mlp_gating")
        
        # Hook each expert
        if 'experts' in mh:
            for i, expert in enumerate(mh['experts']):
                self._register_module_hooks(expert, f"mlp_expert_{i:02d}")

    def _register_attention_enhanced_hooks(self, mlp_root):
        """Register hooks for attention_enhanced MLP architecture."""
        mh = mlp_root.mlp_head
        
        # Feature extractor hooks
        if hasattr(mh, 'feature_extractor'):
            if len(mh.feature_extractor) > 0:
                self._register_module_hooks(mh.feature_extractor[0], "mlp_fe0")
            if len(mh.feature_extractor) > 2:
                self._register_module_hooks(mh.feature_extractor[2], "mlp_fe2")
        
        # Attention hooks
        if hasattr(mh, 'attention'):
            def _attn_hook(mod, inputs, output):
                y = output[0] if isinstance(output, tuple) else output
                if isinstance(y, torch.Tensor):
                    yv = self._to_vec(y)
                    self.cache["mlp_attn_post"] = self._to_np(yv)
            self._hooks.append(mh.attention.register_forward_hook(_attn_hook))
        
        # Post-attention stack
        if hasattr(mh, 'post_attention'):
            post = mh.post_attention
            idx_map = {0: "mlp_post_00_ln", 1: "mlp_post_01_linear", 3: "mlp_post_03_ln",
                      5: "mlp_post_05_linear", 7: "mlp_post_07_linear", 9: "mlp_post_09_linear"}
            
            for i_idx, tag in idx_map.items():
                if i_idx < len(post):
                    if i_idx == 9:  # Final layer, only post hook
                        def _final_hook(mod, inputs, output):
                            y = output[0] if isinstance(output, tuple) else output
                            if isinstance(y, torch.Tensor):
                                self.cache[f"{tag}_post"] = self._to_np(y)
                        self._hooks.append(post[i_idx].register_forward_hook(_final_hook))
                    else:
                        self._register_module_hooks(post[i_idx], tag)

    def _register_generic_mlp_hooks(self, mlp_root):
        """Register hooks for standard/generic MLP architectures."""
        if hasattr(mlp_root, 'mlp_head'):
            self._register_module_hooks(mlp_root.mlp_head, "mlp_generic")

    def get_expected_keys(self) -> List[str]:
        """Get the expected keys for the current model architecture."""
        mlp_root = self.model.mlp_head
        mlp_type = getattr(mlp_root, 'mlp_type', 'unknown')
        
        keys = self.BASE_ENCODER_KEYS.copy()
        
        if mlp_type == "mixture_of_experts":
            keys.extend(self.MOE_KEYS)
        elif mlp_type == "attention_enhanced":
            keys.extend(self.ATTENTION_ENHANCED_KEYS)
        else:
            keys.extend(self.GENERIC_KEYS)
            
        return keys

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def run_once(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Forward once to populate cache; tensors must already be on device.
        Uses inference_mode for better GPU utilization.
        """
        # Ensure tensors are on correct device
        if not hasattr(torch, '_printed_dev'):
            print(f'Input tensors device: {rho_real.device}')
            torch._printed_dev = True
            
        self.register_hooks()
        try:
            with torch.inference_mode():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = self.model(rho_real, rho_imag)  # triggers hooks
                else:
                    _ = self.model(rho_real, rho_imag)  # triggers hooks
                    
            # Get expected keys for this architecture
            expected_keys = self.get_expected_keys()
            
            # Check which expected keys are missing vs which we actually got
            missing = [k for k in expected_keys if k not in self.cache]
            extra = [k for k in self.cache.keys() if k not in expected_keys]
            
            if missing:
                print(f"⚠️ Missing expected keys: {missing}")
                print(f"   Available keys: {list(self.cache.keys())}")
            if extra:
                print(f"   Extra keys found: {extra}")
                
            # Return all available keys that match expected pattern
            result = {}
            for k in expected_keys:
                if k in self.cache:
                    result[k] = self.cache[k]
            
            # Also include any extra keys we found
            for k in extra:
                result[k] = self.cache[k]
                    
            return result
        finally:
            self.remove_hooks()
    
    def time_single_batch(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> float:
        """Time a single batch to verify GPU utilization."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.inference_mode():
            _ = self.model(rho_real, rho_imag)
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time() - t0
