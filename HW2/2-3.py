
import numpy as np
import cv2
import myista
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity



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



img = cv2.imread('im.jpg', 0)
'''img = np.float32(myimg)'''
[h, w] = img.shape

imgmax = img.max()
imgmin = img.min()
for i in range(h):
    for j in range(w):
        img[i, j] = (img[i, j] - imgmin) * 255 / float((imgmax - imgmin))

sigma1 = 0.05*255
sigma2 = 0.1*255
n1 = np.random.normal(0, sigma1, [h, w])
n2 = np.random.normal(0, sigma2, [h, w])
img1 = img + n1
img2 = img + n2

plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(img, cmap=plt.cm.gray), plt.title('original image')
plt.subplot(1, 2, 2), plt.imshow(img1, cmap=plt.cm.gray), plt.title('noisy image')
'''plt.subplot(2, 2, 3), plt.imshow(img2, cmap=plt.cm.gray), plt.title('noisy image 2')'''

h1d = myDCT_basis_gen.myDCT_basis_gen(8)
for i in range(8):
    for j in range(8):
        if i == 0 | j == 0:
            h2d = (np.outer(h1d[:, i], h1d[:, j])).reshape(64, 1)
        else:
            h2d = np.hstack((h2d, (np.outer(h1d[:, i], h1d[:, j])).reshape(64, 1)))

m = h/8
n = w/8
lambda1 = 1
lambda2 = 10

vs1 = np.vsplit(img1, m)
for i in range(m):
    vhs1 = np.hsplit(vs1[i], n)
    for j in range(n):
        y1 = vhs1[j]
        y1 = np.reshape(y1, [64, 1])
        x11 = myista.myista(y1, h2d, lambda1)
        x12 = myista.myista(y1, h2d, lambda2)
        y11 = np.dot(h2d, x11)
        y12 = np.dot(h2d, x12)
        y11 = np.reshape(y11, [8, 8])
        y12 = np.reshape(y12, [8, 8])
        if j == 0:
            img11row = y11
            img12row = y12
        else:
            img11row = np.hstack((img11row, y11))
            img12row = np.hstack((img12row, y12))
    if i == 0:
        img11 = img11row
        img12 = img12row
    else:
        img11 = np.vstack((img11, img11row))
        img12 = np.vstack((img12, img12row))

vs2 = np.vsplit(img2, m)
for i in range(m):
    vhs2 = np.hsplit(vs2[i], n)
    for j in range(n):
        y2 = vhs2[j]
        y2 = np.reshape(y2, [64, 1])
        x21 = myista.myista(y2, h2d, lambda1)
        x22 = myista.myista(y2, h2d, lambda2)
        y21 = np.dot(h2d, x21)
        y22 = np.dot(h2d, x22)
        y21 = np.reshape(y21, [8, 8])
        y22 = np.reshape(y22, [8, 8])
        if j == 0:
            img21row = y21
            img22row = y22
        else:
            img21row = np.hstack((img21row, y21))
            img22row = np.hstack((img22row, y22))
    if i == 0:
        img21 = img21row
        img22 = img22row
    else:
        img21 = np.vstack((img21, img21row))
        img22 = np.vstack((img22, img22row))


plt.show()

img11 = rescale_intensity(img11, in_range=(0,255))

img11 = (img11 * 255).astype("uint8")

cv2.namedWindow("original")
cv2.imshow("original",img11)


cv2.waitKey(0)
cv2.destroyAllWindows()

'''plt.figure(2)
plt.imshow(img11)'''
'''plt.subplot(2, 2, 1), plt.imshow(img11, cmap=plt.cm.gray), plt.title('recovered image(noise level 1, lambda=1', fontsize = 8)
plt.subplot(2, 2, 2), plt.imshow(img12, cmap=plt.cm.gray), plt.title('recovered image(noise level 1, lambda=10)', fontsize = 8)
plt.subplot(2, 2, 3), plt.imshow(img21, cmap=plt.cm.gray), plt.title('recovered image(noise level 2, lambda=1)', fontsize = 8)
plt.subplot(2, 2, 4), plt.imshow(img22, cmap=plt.cm.gray), plt.title('recovered image(noise level 2, lambda=10)', fontsize = 8)'''

