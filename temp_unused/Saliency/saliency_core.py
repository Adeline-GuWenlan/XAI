# xai_core/saliency_core.py
import torch

def interpolate_inputs(baseline, inputs, steps):
    alphas = torch.linspace(0, 1, steps).view(-1, *([1] * (len(inputs.shape) - 1)))
    return baseline + alphas * (inputs - baseline)

def compute_integrated_gradients(model, inputs, baseline, steps=50, target=None):
    interpolated_inputs = interpolate_inputs(baseline, inputs, steps)
    gradients = []

    for i in range(steps):
        inp = interpolated_inputs[i].unsqueeze(0)
        inp.requires_grad_(True)

        output = model(inp)

        if target is not None:
            output = output[0, target]
        else:
            output = output.sum()

        grad = torch.autograd.grad(outputs=output, inputs=inp, create_graph=False, retain_graph=False)[0]
        gradients.append(grad)

    avg_grad = torch.mean(torch.stack(gradients), dim=0)
    integrated_grads = (inputs - baseline) * avg_grad
    return integrated_grads
