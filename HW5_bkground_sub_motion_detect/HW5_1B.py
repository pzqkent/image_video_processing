
import numpy as np

import imutils
import cv2


cap = cv2.VideoCapture('Q1.avi')

fgbg = cv2.BackgroundSubtractorMOG()

numframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
count = 1
firstFrame = None

while count <= 100:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue

    name = 'Frame%s.jpg' %count

    fgmask = fgbg.apply(frame)
    frame = imutils.resize(frame, width=500)


    frameDelta = cv2.absdiff(firstFrame, gray)
    # frameDelta = cv2.absdiff(frame,frame)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # if the contour is too small, ignore it
        # if cv2.contourArea(c) < args["min_area"]:
        # 	continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print numframes
    # cv2.imwrite(name,fgmask)
    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    count += 1
cap.release()


# firstFrame = None
# frameDelta = cv2.absdiff(firstFrame, gray)
# thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.dilate(thresh, None, iterations=2)
# (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in cnts:
# 	# if the contour is too small, ignore it
# 	# if cv2.contourArea(c) < args["min_area"]:
# 	# 	continue
# 	# compute the bounding box for the contour, draw it on the frame,
# 	# and update the text
#
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



# while (1):
#
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#     print frame
#
#     cv2.imshow('frame', fgmask)
#     #
#     # k = cv2.waitKey(30) & 0xff
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     #
#     # if k == 27:
#     #
#     #     break
#
#
# cap.release()
#
# cv2.destroyAllWindows()