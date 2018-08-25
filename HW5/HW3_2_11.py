import cv2
import matplotlib .pyplot as plt

def getSift():

    img_path1 = 'Frame_q2_1.jpg'

    img = cv2.imread(img_path1)

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()

    kp = sift.detect(gray,None)
    print type(kp),type(kp[0])

    print kp[0].pt

    des = sift.compute(gray,kp)
    print type(kp),type(des)

    print type(des[0]), type(des[1])
    print des[0],des[1]

    print des[1].shape

    img=cv2.drawKeypoints(gray,kp)
    #cv2.imwrite('sift_keypoints.jpg',img)
    plt.imshow(img),plt.title('sift_keypoints', fontsize =20),plt.show()
getSift()
