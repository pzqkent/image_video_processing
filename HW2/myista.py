import numpy as np


def soft(x, T):
    [M, N] = x.shape
    for i in range(M):
        for j in range(N):
            if x[i, j] < - T:
                x[i, j] = x[i, j] + T
            elif x[i, j] > T:
                x[i, j] = x[i, j] - T
            else:
                x[i, j] = 0
    return x


def myista(y, H, lamda):
    old_x = 0 * np.dot(H.T, y)

    [w, v] = np.linalg.eig(np.dot(H.T, H))
    alpha = w.max()
    Tres = lamda / (2 * alpha)

    new_x = soft(old_x + np.dot(H.T, (y - np.dot(H, old_x))) / alpha, Tres)
    old_error = np.dot((y-np.dot(H, old_x)).T, (y - np.dot(H, old_x))) + lamda * sum(abs(old_x))
    new_error = np.dot((y-np.dot(H, new_x)).T, (y - np.dot(H, new_x))) + lamda * sum(abs(new_x))
    error_ratio = (old_error - new_error) / old_error
    old_error = new_error
    old_x = new_x

    while error_ratio > 1e-3:
        new_x = soft(old_x + np.dot(H.T, (y - np.dot(H, old_x))) / alpha, Tres)
        old_error = np.dot((y - np.dot(H, old_x)).T, (y - np.dot(H, old_x))) + lamda * sum(abs(old_x))
        new_error = np.dot((y - np.dot(H, new_x)).T, (y - np.dot(H, new_x))) + lamda * sum(abs(new_x))
        error_ratio = (old_error - new_error) / old_error
        old_error = new_error
        old_x = new_x

    return new_x