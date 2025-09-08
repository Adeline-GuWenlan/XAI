# competitive_loss.py
# Core competitive MoE losses & helpers
# ---------------------------------------------------------
# Expected shapes
# - gates:        (B, E)           softmax probabilities from the gate
# - expert_outs:  (B, 1, E)        per-expert predictions (same target dim)
# - y:            (B, 1)           target
#
# You can switch Gaussian→Laplace by swapping the log-likelihood.

import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def entropy_from_probs(pi, eps=1e-9):
    """Mean Shannon entropy of (B,E) gate probs."""
    p = pi.clamp_min(eps)
    return (-p * p.log()).sum(dim=-1).mean()

def _apply_temperature_and_topk(pi, tau=None, topk=None, eps=1e-9):
    """Optionally sharpen/soften the gate *inside the loss only*."""
    if tau is not None:
        log_pi = (pi.clamp_min(eps)).log() / tau
        pi = F.softmax(log_pi, dim=-1)

    if topk is not None and 1 <= topk < pi.shape[-1]:
        topv, topi = pi.topk(topk, dim=-1)
        mask = torch.zeros_like(pi).scatter(1, topi, 1.0)
        pi = (pi * mask)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)

    return pi

def mixture_gaussian_nll(gates, expert_outs, y, sigma=0.1, eps=1e-9, tau=None, topk=None):
    """
    Competitive mixture NLL:
      L = - mean_i log sum_j pi_ij * N(y_i | f_j(x_i), sigma^2)

    Args:
      gates:       (B, E) probabilities
      expert_outs: (B, 1, E)
      y:           (B, 1)
    """
    pi = _apply_temperature_and_topk(gates, tau=tau, topk=topk, eps=eps)  # (B,E)

    # log N(y | mu, sigma^2)
    diff  = y.unsqueeze(-1) - expert_outs                # (B,1,E)
    log_p = -0.5 * (diff**2) / (sigma**2) \
            - math.log(sigma) - 0.5 * math.log(2*math.pi)  # (B,1,E)

    log_pi  = (pi.clamp_min(eps)).log().unsqueeze(1)     # (B,1,E)
    log_mix = torch.logsumexp(log_pi + log_p, dim=-1)    # (B,1)
    return -(log_mix.mean())

def mixture_laplace_nll(gates, expert_outs, y, b=0.1, eps=1e-9, tau=None, topk=None):
    """
    Optional Laplace likelihood:
      p(y|mu) = (1/(2b)) * exp(-|y-mu|/b)
    """
    pi = _apply_temperature_and_topk(gates, tau=tau, topk=topk, eps=eps)  # (B,E)
    abs_diff = (y.unsqueeze(-1) - expert_outs).abs()       # (B,1,E)
    log_p = -(abs_diff / b) - math.log(2*b)                # (B,1,E)
    log_pi = (pi.clamp_min(eps)).log().unsqueeze(1)        # (B,1,E)
    log_mix = torch.logsumexp(log_pi + log_p, dim=-1)      # (B,1)
    return -(log_mix.mean())

def balance_kl(gates, eps=1e-9):
    """
    Load-balancing penalty: KL(usage || uniform), where usage is
    the soft mean over the batch.
    """
    u = gates.mean(dim=0).clamp_min(eps)   # (E,)
    E = u.shape[0]
    return (u * (u.log() - math.log(1.0/E))).sum()

def effective_num_experts(gates, eps=1e-9):
    """
    N_eff = exp( H( mean over batch of gate probs ) ).
    Higher is more balanced usage across experts.
    """
    u = gates.mean(dim=0).clamp_min(eps)   # (E,)
    H = -(u * u.log()).sum()
    return torch.exp(H)
