
import numpy as np

import imutils
import cv2


cap = cv2.VideoCapture('Q1.avi')

fgbg = cv2.BackgroundSubtractorMOG(history=1,nmixtures = 1, backgroundRatio = 0.7, noiseSigma = 0 )

numframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
count = 1

while count <= 100:

    name = 'Frame%s.jpg' %count
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)



    # print numframes
    # cv2.imwrite(name,fgmask)
    cv2.imshow('frame', fgmask)
    cv2.waitKey(100)
    count += 1
cap.release()


