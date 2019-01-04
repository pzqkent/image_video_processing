import numpy as np
import myWaveletTransform


def soft(x, T):
    [m, n] = x.shape
    for i in range(m):
        for j in range(n):
            if x[i, j] < - T:
                x[i, j] = x[i, j] + T
            elif x[i, j] > T:
                x[i, j] = x[i, j] - T
            else:
                x[i, j] = 0
    return x


def myista2(y, wavelet, lamda):
    old_x = 0 * y

    Tres = lamda / float(2 * 50)

    new_x = soft(old_x + myWaveletTransform.forward_transform(y - myWaveletTransform.inverse_transform(old_x, wavelet), wavelet)/50, Tres)
    old_cre = y - myWaveletTransform.inverse_transform(old_x, wavelet)
    old_cre = old_cre.reshape(old_cre.size, 1)
    old_error = np.dot(old_cre.T, old_cre) + lamda * sum(sum(abs(old_x)))
    new_cre = y - myWaveletTransform.inverse_transform(new_x, wavelet)
    new_cre = new_cre.reshape(new_cre.size, 1)
    new_error = np.dot(new_cre.T, new_cre) + lamda * sum(sum(abs(new_x)))
    error_ratio = (old_error - new_error) / old_error
    error = np.log(old_error)
    old_error = new_error
    old_x = new_x

    while error_ratio > 1e-3:
        new_x = soft(old_x + myWaveletTransform.forward_transform(y - myWaveletTransform.inverse_transform(old_x, wavelet), wavelet)/50, Tres)
        old_cre = y - myWaveletTransform.inverse_transform(old_x, wavelet)
        old_cre = old_cre.reshape(old_cre.size, 1)
        old_error = np.dot(old_cre.T, old_cre) + lamda * sum(sum(abs(old_x)))
        new_cre = y - myWaveletTransform.inverse_transform(new_x, wavelet)
        new_cre = new_cre.reshape(new_cre.size, 1)
        new_error = np.dot(new_cre.T, new_cre) + lamda * sum(sum(abs(new_x)))
        error_ratio = (old_error - new_error) / old_error
        error = np.hstack((error, np.log(old_error)))
        old_error = new_error
        old_x = new_x

    return new_x, error
