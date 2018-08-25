import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt


def forward_transform(y, wavelet):
    [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(y, wavelet, 'per', 3)
    cA2 = np.vstack((np.hstack((cA3, cH3)), np.hstack((cV3, cD3))))
    cA1 = np.vstack((np.hstack((cA2, cH2)), np.hstack((cV2, cD2))))
    return np.vstack((np.hstack((cA1, cH1)), np.hstack((cV1, cD1))))


def inverse_transform(x, wavelet):
    cA1 = np.vsplit(np.hsplit(x, 2)[0], 2)[0]
    cV1 = np.vsplit(np.hsplit(x, 2)[0], 2)[1]
    cH1 = np.vsplit(np.hsplit(x, 2)[1], 2)[0]
    cD1 = np.vsplit(np.hsplit(x, 2)[1], 2)[1]
    cA2 = np.vsplit(np.hsplit(cA1, 2)[0], 2)[0]
    cV2 = np.vsplit(np.hsplit(cA1, 2)[0], 2)[1]
    cH2 = np.vsplit(np.hsplit(cA1, 2)[1], 2)[0]
    cD2 = np.vsplit(np.hsplit(cA1, 2)[1], 2)[1]
    cA3 = np.vsplit(np.hsplit(cA2, 2)[0], 2)[0]
    cV3 = np.vsplit(np.hsplit(cA2, 2)[0], 2)[1]
    cH3 = np.vsplit(np.hsplit(cA2, 2)[1], 2)[0]
    cD3 = np.vsplit(np.hsplit(cA2, 2)[1], 2)[1]
    return pywt.waverec2([cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)], wavelet, 'per')





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




img = cv2.imread('lena_gray.bmp', 0)
[h, w] = img.shape

imgmax = img.max()
imgmin = img.min()
for i in range(h):
    for j in range(w):
        img[i, j] = (img[i, j] - imgmin) * 255 / float((imgmax - imgmin))

n = np.random.normal(0, 0.05*255, [h, w])
img_n = img + n

x_Haar1 = myWaveletTransform.forward_transform(img_n, 'Haar')
x_Daub1 = myWaveletTransform.forward_transform(img_n, 'db8')

x_Haar2, error_Haar = myista2.myista2(img_n, 'haar', 10)
x_Daub2, error_Daub = myista2.myista2(img_n, 'db8', 10)

n_Haar = np.linspace(0, error_Haar.size-1, error_Haar.size)
n_Daub = np.linspace(0, error_Daub.size-1, error_Daub.size)

y_Haar = myWaveletTransform.inverse_transform(x_Haar2, 'haar')
y_Daub = myWaveletTransform.inverse_transform(x_Daub2, 'db8')

plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(img, cmap=plt.cm.gray), plt.title('original image')
plt.subplot(1, 2, 2), plt.imshow(img_n, cmap=plt.cm.gray), plt.title('noisy image')
plt.figure(2)
plt.subplot(1, 2, 1), plt.imshow(x_Haar1, cmap=plt.cm.gray), plt.title('Haar Transform(noisy image)')
plt.subplot(1, 2, 2), plt.imshow(x_Daub1, cmap=plt.cm.gray), plt.title('Daubichies Transform(noisy image)')
plt.figure(3)
plt.subplot(1, 2, 1), plt.imshow(x_Haar2, cmap=plt.cm.gray), plt.title('Haar Transform(denoised image)', fontsize = 8)
plt.subplot(1, 2, 2), plt.imshow(x_Daub2, cmap=plt.cm.gray), plt.title('Daubichies Transform(denoised image)', fontsize = 8)
plt.figure(4)
plt.subplot(1, 2, 1), plt.imshow(y_Haar, cmap=plt.cm.gray), plt.title('denoised image(Haar)')
plt.subplot(1, 2, 2), plt.imshow(y_Daub, cmap=plt.cm.gray), plt.title('denoised image(Daub)')
plt.figure(5)
plt.subplot(2, 1, 1), plt.plot(n_Haar, error_Haar.reshape(error_Haar.size,)), plt.title('error vs iteration(Haar)')
plt.subplot(2, 1, 2), plt.plot(n_Daub, error_Daub.reshape(error_Daub.size,)), plt.title('error vs iteration(Daub)')
plt.show()
