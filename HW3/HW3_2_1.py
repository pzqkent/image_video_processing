import cv2
import matplotlib .pyplot as plt

def getSift():

    img_path1 = 'im.jpg'

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
    plt.imshow(img),plt.show()
getSift()

def matchSift3():

    img1 = cv2.imread('im.jpg', 0)  # queryImage
    img2 = cv2.imread('im90.jpg', 0)  # trainImage
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.drawMatchesKnn expects list of lists as matches.

    # Apply ratio test

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.line(img1, kp1, img2, kp2, good[:10], None)

    plt.imshow(img3), plt.show()



def matchSift():

    img1 = cv2.imread('im.jpg', 0)  # queryImage
    img2 = cv2.imread('im45.jpg', 0)  # trainImage
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.drawMatchesKnn expects list of lists as matches.

    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)
    explore_match('find_obj', img1, img2, kp_pairs)  # cv2 shows image
    cv2.waitKey()
    cv2.destroyAllWindows()

matchSift()