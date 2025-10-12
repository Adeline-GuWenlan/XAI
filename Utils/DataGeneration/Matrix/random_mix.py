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
    Random mixed state: mix several random pure states using random weights.
    rank: number of pure states involved in the mixture (≤ D). Defaults to D.
    """
    # modified 9.9 - 
    # change the distribution: probabilities p multiples a pure state and (1-p)identity/ divode by d
    p = random.random()
    rho = random_pure_state(D) * p + (1 - p) * np.eye(D) / D
    print("thse is rho before normalization:")
    print(rho)
    print("this is trace before normalization:")
    print(np.trace(rho))

    return rho  # Normalize to ensure trace is 1
    """rank = D if rank is None else min(rank, D)
    weights = np.random.rand(rank)
    weights /= weights.sum()
    rho = sum(w * random_pure_state(D) for w in weights)
    return rho"""
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
    print(rho)
    print(rho.conj())

    if np.linalg.norm(rho - rho.conj().T, ord=2) > eps:
        print()
        return False, "ρ is not Hermitian"

    # Trace
    trace_rho = np.trace(rho)
    trace_rho_conj = np.trace(rho.conj())
    print(trace_rho, trace_rho_conj)
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
