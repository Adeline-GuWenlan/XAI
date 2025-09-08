
import numpy as np
# helper functions

si = np.array([[1,0], [0,1]])
sx = np.array([[0,1] ,[1,0]])
sy = np.array([[0,-1j], [1j,0]])
sz = np.array([[1,0], [0,-1]])

pauli = [si, sx, sy, sz]

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# stabilizer norm (density matrix -> real number)

def get_sn(rho):

    p = int(np.log2(np.size(rho[0])))
    dim = 2**p
    dim2 = 4**p
    
    a0 = 0
    for no in range(dim2):
        ntb = numberToBase(no, 4)
        op_no = np.pad(ntb, (p-len(ntb), 0), 'constant')
        op = [[1]]
        for i in range(p):
            op = np.kron(op,pauli[op_no[i]])
        a0 = np.array(a0+op)
    a0 = a0/dim
    
    aulist = []
    for no in range(dim2):
        ntb = numberToBase(no, 4)
        op_no = np.pad(ntb, (p-len(ntb), 0), 'constant')
        op = [[1]]
        for i in range(p):
            op = np.kron(op,pauli[op_no[i]])
        aulist.append( np.dot(np.dot(op, a0), np.matrix(op).getH()) )
    
    wigner = [np.trace(np.dot(aulist[n], rho))/dim for n in range(dim2)]
    return np.real(np.sum(np.absolute(wigner)-wigner)/2)

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
