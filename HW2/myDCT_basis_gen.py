import numpy as np


def myDCT_basis_gen(N):
    alpha = np.zeros(N)
    H = np.zeros((N, N))
    for k in range(N):
        if k == 0:
            alpha[k] = np.sqrt(1/float(N))
        else:
            alpha[k] = np.sqrt(2/float(N))
        for n in range(N):
            H[n, k] = alpha[k] * np.cos((2*n+1)*k*np.pi/float(2*N))
    return H