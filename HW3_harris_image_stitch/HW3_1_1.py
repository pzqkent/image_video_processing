
import cv2
import numpy as np
import matplotlib
import math
from matplotlib import pyplot as plt


# filename = 'im.jpg'
# tu = cv2.imread(filename)


# gray = cv2.cvtColor(tu, cv2.COLOR_RGB2GRAY)
gray = cv2.imread('4.1.05.tiff',cv2.IMREAD_GRAYSCALE)


print 'get the size of the image'
print gray.shape
print '\n'





#print gray[:2,:2]


m = np.matrix(gray)


delta_h = m
def grad_x(h):
    a = int(h.shape[0])
    b = int(h.shape[1])

    for i in range(a):
        for j in range(b):
            if i-1>=0 and i+1<a and j-1>=0 and j+1<b:
                c = abs(int(h[i-1,j-1]) - int(h[i+1,j-1]) + 2*(int(h[i-1,j]) - int(h[i+1,j])) + int(h[i-1,j+1]) - int(h[i+1,j+1]))
#                print c
                if c>255:
#                    print c
                    c = 255
                delta_h[i,j] = c
            else:
                delta_h[i,j] = 0
    print 'gradient in x: %s' %delta_h
    return delta_h


def grad_y(h):
    a = int(h.shape[0])
    b = int(h.shape[1])

    for i in range(a):
        for j in range(b):
            if i-1>=0 and i+1<a and j-1>=0 and j+1<b:
                c = abs(int(h[i-1,j-1]) - int(h[i-1,j+1]) + 2*(int(h[i,j-1]) - int(h[i,j+1])) + (int(h[i+1,j-1]) - int(h[i+1,j+1])))
#                print c
                if c > 255:
                    c = 255
                delta_h[i,j] = c
            else:
                delta_h[i,j] = 0
    print 'gradient in y:%s' %delta_h
    return delta_h


img_laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

dx = np.array(grad_x(gray))
dy = np.array(grad_y(gray))

#dxy = dx + dy
#print 'dxy1:'
#print dxy

A = dx * dx
B = dy * dy
C = dx * dy

print A
print B
print C

A1 = A
B1 = B
C1 = C

A1 = cv2.GaussianBlur(A1,(3,3),1.5)
B1 = cv2.GaussianBlur(B1,(3,3),1.5)
C1 = cv2.GaussianBlur(C1,(3,3),1.5)

print A1
print B1
print C1

a = int(gray.shape[0])
b = int(gray.shape[1])

R = np.zeros(gray.shape)
for i in range(a):
    for j in range(b):
        M = [[A1[i,j],C1[i,j]],[C1[i,j],B1[i,j]]]

        R[i,j] = np.linalg.det(M) - 0.06 * (np.trace(M)) * (np.trace(M))

print 'R:'
print R


import matplotlib .pyplot as plt



from skimage.feature import peak_local_max
coordinates = peak_local_max(R, min_distance =10)




plt.imshow(gray,cmap="gray")
# plt.autoscale (False)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.title('Peak local maxima', fontsize =20)
plt.axis('off')
plt.show()

print 'coordinates:'
print coordinates
# cv2.namedWindow('R',cv2.WINDOW_NORMAL)
# cv2.imshow('R',coordinates)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from skimage.feature import peak_local_max