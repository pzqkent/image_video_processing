import numpy as np
from scipy import signal
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

k =[]
size = 3
for x in range(0,size):
    k.append([1.0]*size)

h,w = signal.freqz(k)
