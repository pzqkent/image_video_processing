import numpy as np
from scipy import ndimage
import cv2
import argparse
import matplotlib.pyplot as plot
from skimage.exposure import rescale_intensity

k =[]
size = input("please enter the size:",)
for x in range(0,size):
    k.append([1]*size)
print k

for x in range(0,size):
    k[x] = input("please enter the %s th row like [1,2,3]:" %(x+1),)


print k
result = ndimage.convolve(k,k,mode = 'constant',cval = 0.0)

print result