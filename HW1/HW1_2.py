from PIL import Image
from pylab import *
from numpy import*
import random


im = array(Image.open('im.jpg'))

means = 0
sigma = 100


r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()




for i in range(im.shape[0]*im.shape[1]):

    pr = int(r[i]) + random.gauss(0,sigma)

    pg = int(g[i]) + random.gauss(0,sigma)

    pb = int(b[i]) + random.gauss(0,sigma)

    if(pr < 0):
        pr = 0

    if(pr > 255):
        pr = 255

    if(pg < 0):
        pg = 0

    if(pg > 255):

        pg = 255

    if(pb < 0):
        pb = 0

    if(pb > 255):
        pb = 255

    r[i] = pr
    g[i] = pg
    b[i] = pb


im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])



imshow(im)

show()
from scipy import misc
misc.imsave('noisy.jpg', im)
