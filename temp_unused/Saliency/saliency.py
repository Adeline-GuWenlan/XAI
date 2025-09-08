# xai_core/saliency.py
import torch
from .saliency_core import compute_integrated_gradients

def compute_saliency(model, inputs, method="integrated_gradients", baseline=None, **kwargs):
    assert method == "integrated_gradients", "Only integrated gradients are supported in this version."
    
    model.eval()
    inputs.requires_grad = True

    # Run IG
    saliency_map = compute_integrated_gradients(
        model=model,
        inputs=inputs,
        baseline=baseline if baseline is not None else torch.zeros_like(inputs),
        steps=50,  # adjustable
        target=None,  # assuming single scalar output
        **kwargs
    )
    return saliency_map.detach().cpu()
