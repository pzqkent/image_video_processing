import cv2
import numpy as np
import imutils


# name = 'ted.jpg'
name = 'Frame_ori10.jpg'

template = cv2.imread(name,0)
template = imutils.resize(template, width=500)
face_w, face_h = template.shape[::-1]

cv2.namedWindow('image')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Q1.avi')

threshold = 1
ret = True
count = 1

while count <= 20 :
    ret, img = cap.read()

    if img is None:
        break;
    #flip the image  ! optional
    # img = cv2.flip(img,1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

    if len(res):
        location = np.where( res >= threshold)
        for pt in zip(*location[::-1]):
            #puting  rectangle on recognized erea
            cv2.rectangle(img, pt, (pt[0] + face_w, pt[1] + face_h), (0,0,255), 2)

    cv2.imshow('image',img)
    k = cv2.waitKey(1000) & 0xFF
    if k == 27:
        break
    count += 1
cv2.destroyAllWindows()