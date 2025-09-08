import numpy as np
def vec_to_rho(vec: np.ndarray) -> np.ndarray:
    """
    反展开：把向量还原成 Hermitian 矩阵。
    """
    # 通过方程 D^2 = len(vec) 求 D
    D = int(np.sqrt(len(vec)))
    if D * D != len(vec):
        raise ValueError("Length of vec must be a perfect square.")
    rho = np.zeros((D, D), dtype=np.complex128)
    k = 0
    for i in range(D):
        rho[i, i] = vec[k]
        k += 1
        for j in range(i + 1, D):
            rho[i, j] = vec[k] + 1j * vec[k + 1]
            rho[j, i] = rho[i, j].conj()
            k += 2
    return rho
