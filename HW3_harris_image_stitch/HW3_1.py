import cv2
import numpy as np
import matplotlib .pyplot as plt
from skimage.feature import corner_peaks
from scipy.ndimage import filters
from matplotlib import pyplot as plt



# gray = cv2.imread('4.1.05.tiff',cv2.IMREAD_GRAYSCALE)
gray = cv2.imread('im_double.jpg',cv2.IMREAD_GRAYSCALE)

def harris_eps(im, sigma=3):
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)

    Wdet = Wxx*Wyy - Wxy**2

    Wtr = Wxx + Wyy

    return Wdet * 2 / (Wtr + 1e-06)

result1 = []
R = harris_eps(gray,sigma=1)
# R = R.tolist()
print 'R'
print R

result0 =[]
my_coords1 = corner_peaks(harris_eps(gray, sigma=1), min_distance=3, threshold_rel=0)
for i in my_coords1:
    print 'ri'
    # print R[i(0),i(1)]
    print 'i'
    print i[0]
    print i[1]
    print i
    print R[i[0],i[1]]


    result1.append(R[i[0],i[1]])
    # result1.append(R[i[:,0],i[:,1]])



print 'result1'
print result1

result2 = sorted(result1,reverse=True)
print 'result2:'
print result2

result3 = result2[0:50]
print 'result3'
print result3
print len(result3)

# result4 = np.zeros((1,len(result3)))
result4 = []
k = 1
for j in result3:
    result4.append(np.argwhere(R == j))
    # result4[k] = np.argwhere(R == j)
    print np.argwhere(R == j)
    k = k + 1

result4 = np.array(result4)
print 'result4'
print result4
print 'result4_11:'
print result4.shape
print result4[1,0,1]

plt.imshow(gray,cmap="gray")
# plt.autoscale (False)
plt.plot(result4[:,0,1], result4[:,0,0], 'ro')
plt.title('Features Detected', fontsize =20)
plt.axis('off')
plt.show()

print 'my_coords1'
print my_coords1

print 'Ri'
print R[1,1]
print R