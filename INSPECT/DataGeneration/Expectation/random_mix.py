# 6.12 update：检查conj是不是对的
# Modified. Pending for check!!!
# weighted pure state + probabilities (weight) to mixed state
# purity less than one (change coefficient) (How to implement this step?)
# the number of pure states doesn't need to exceed the dimensions
import random
import numpy as np
def random_pure_state(D: int) -> np.ndarray:
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())
# Start from the smallest matrix to construct a pure state - 
# random valid (random complex matrix), normalize (Did I implement this step?), 
# dot product (Is this step necessary??) + random weights

def random_mixed_state(D: int) -> np.ndarray:
    """
    Random mixed state using convex combination method.
    Combines a random pure state with maximally mixed state.

    Args:
        D: Dimension of the Hilbert space

    Returns:
        Density matrix ρ = p|ψ⟩⟨ψ| + (1-p)I/D where p ~ Uniform(0,1)
    """
    # modified 9.9 -
    # change the distribution: probabilities p multiples a pure state and (1-p)identity/ divode by d
    p = random.random()
    rho = random_pure_state(D) * p + (1 - p) * np.eye(D) / D
    # Normalize trace to exactly 1
    rho = rho / np.trace(rho)
    return rho


def random_mixed_state_poisson(D: int, lam: float = 1.0) -> np.ndarray:
    """
    Random mixed state using Poisson-distributed number of pure states.
    More control over mixing via lambda parameter.

    Args:
        D: Dimension of the Hilbert space
        lam: Lambda parameter for Poisson distribution (controls mixing level)
             - Small lam (e.g., 0.5): mostly single pure states (less mixed)
             - Large lam (e.g., 5.0): many pure states mixed (more mixed)

    Returns:
        Density matrix ρ = (1/K) Σ |ψᵢ⟩⟨ψᵢ| where K ~ Poisson(lam) + 1
    """
    # K ~ Poisson(lam) + 1 ensures at least 1 pure state
    K = np.random.poisson(lam) + 1

    # Initialize empty density matrix
    rho = np.zeros((D, D), dtype=np.complex128)

    # Mix K random pure states with equal weights
    for _ in range(K):
        rho += random_pure_state(D)

    # Average over K states
    rho /= K

    # Normalize trace to exactly 1
    rho /= np.trace(rho)

    return rho
def _tolerance(dim, base=1e-12, scale='linear'):
    """
    Return a tolerance threshold based on the given dimension.
      - base  : absolute precision used for small dimensions
      - scale : 'linear' uses base * dim; 'sqrt' uses base * np.sqrt(dim)
    """
    if scale == 'linear':
        return base * dim           # 最稳妥保险
    elif scale == 'sqrt':
        return base * np.sqrt(dim)  # 更贴近平均增长
    else:
        raise ValueError("scale must be 'linear' or 'sqrt'")


def validate_rho(rho, base_eps=1e-12, scale='linear'):
    dim = rho.shape[0]
    eps = _tolerance(dim, base_eps, scale)

    # Hermitian
    if np.linalg.norm(rho - rho.conj().T, ord=2) > eps:
        return False, "ρ is not Hermitian"

    # Trace
    tr = np.trace(rho)
    if abs(tr - 1) > eps:
        return False, f"Trace={tr}"

    # PSD
    eigvals = np.linalg.eigvalsh(rho)
    if eigvals.min() < -eps:
        return False, f"Neg. eigval {eigvals.min()}"

    return True, "ρ valid"



"""trace(rho)
eigenvalues (should be positive) and eigenvectors of rho
Hermitian"""
# can ve used to generate random mixed states for the input training data.

'''if __name__ == "__main__":
    rho = random_mixed_state(D=4, rank=3)
    print("Generated random mixed state (ρ):")
    ok, msg = validate_rho(rho)
    print(f"Validation result: {ok}, Message: {msg}")
'''
