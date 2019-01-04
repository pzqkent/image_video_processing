import myista
import myDCT_basis_gen
import numpy as np


x = np.array([(0, 0, 1, 0)])
x = x.T
H = myDCT_basis_gen.myDCT_basis_gen(4)
y = np.dot(H, x)
print myista.myista(y, H, 0.1)
