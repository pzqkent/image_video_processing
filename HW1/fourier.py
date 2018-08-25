import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('im.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')
plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')
plt.show()