import cv2


im = cv2.imread('im.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('original',im)



#im_lowers = cv2.pyrDown(im)
#cv2.imshow('im_lowers',im_lowers)


s = cv2.SIFT()

keypoints = s.detect(im)


for k in keypoints:
    cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)
    #cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)


cv2.imshow('SIFT_features',im)
cv2.waitKey()
cv2.destroyAllWindows()

print keypoints