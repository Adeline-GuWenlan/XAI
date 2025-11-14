import numpy as np

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

def get_explist(rho):

    p = int(np.log2(np.size(rho[0])))
    dim2 = 4**p
    
    explist = []
    for no in range(dim2):
        ntb = numberToBase(no, 4)
        opbin = np.pad(ntb, (p-len(ntb), 0), 'constant')
        op = [[1]]
        for i in range(p):
            op = np.kron(op,pauli[opbin[i]])
        explist.append(np.trace(np.dot(rho,op)))

    explist = np.real(explist)
    explist[np.abs(explist) < 1e-8] = 0
    return np.real(explist)