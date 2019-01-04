import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

def convolution_2d(a,b):
    output1 = ndimage.convolve(myimg1,k,mode = 'constant',cval = 0.0)
    return output1


myimg = cv2.imread('noisy.jpg',0)
myimg1 = np.float32(myimg)
print "myimg1:",myimg

'''cv2.namedWindow("Image")
cv2.imshow("Image",myimg)
cv2.waitKey(0)'''

'''filter_size = int(raw_input("please enter the filter size:"))'''

'''filter_coefficients = raw_input("filter coefficients:")
filter_coefficients = int(filter_coefficients)'''

'''k = np.array(int(input("input the filter:")))
print k'''

k =[]
size = input("please enter the filter size N:",)
for x in range(0,size):
    k.append([1.0]*size)


for x in range(0,size):
    k[x] = input("please enter the %s th row like [1,2,3]:" %(x+1),)
print "the filter you entered is:", k



'''result = ndimage.convolve(myimg1,k,mode = 'constant',cval = 0.0)'''

result = convolution_2d(myimg1,k)
print result

result = rescale_intensity(result, in_range=(0,255))

result = (result * 255).astype("uint8")

print result
cv2.namedWindow("original")
cv2.imshow("original",myimg)
cv2.namedWindow("result")
cv2.imshow('result',result)

cv2.waitKey(0)
cv2.destroyAllWindows()


original = np.fft.fft2(myimg)
transformed = np.fft.fft2(result)
original_central = np.fft.fftshift(original)
transformed_central = np.fft.fftshift(transformed)
filterfre = np.fft.fft2(k,s=(len(myimg1),len(myimg1)))
filterfre_central = np.fft.fftshift(filterfre)

s1 = np.log(np.abs(original_central))
s2 = np.log(np.abs(transformed_central))
s3 = np.log(np.abs(filterfre_central))

plt.subplot(131),plt.imshow(s1,'gray'),plt.title('original image')
plt.subplot(132),plt.imshow(s2,'gray'),plt.title('filtered image')
plt.subplot(133),plt.imshow(s3,'gray'),plt.title('the filter')
plt.show()

