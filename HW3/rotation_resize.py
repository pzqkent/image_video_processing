import cv2

gray = cv2.imread('im.jpg',cv2.IMREAD_GRAYSCALE)
height = gray.shape[0]
width = gray.shape[1]
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),90,1)
output1 = cv2.warpAffine(gray,rotation_matrix,(width,height))


# cv2.namedWindow("Image_rotate_90")
# cv2.imshow('Image_rotate_90',output1)
# cv2.waitKey (0)
# cv2.imwrite('im45.jpg', output1)

# im_half = np.zeros((width/2,height/2), np.uint8)
im_half = cv2.resize(gray,(width/2,height/2))
cv2.imwrite('im_half.jpg',im_half)

im_double = cv2.resize(gray,(width*2,height*2))
cv2.imwrite('im_double.jpg',im_double)

cv2.namedWindow("Image_Half")
cv2.imshow('Image_Double',im_double)
cv2.waitKey (0)
