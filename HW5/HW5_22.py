import cv2
import numpy as np
import matplotlib.pyplot as plt

fgbg = cv2.BackgroundSubtractorMOG()
vid = cv2.VideoCapture("Q2.avi")

Framerate = vid.get(cv2.cv.CV_CAP_PROP_FPS)

ret, frame = vid.read()
f1 = frame

ret, frame = vid.read()
ret, frame = vid.read()
ret, frame = vid.read()
ret, frame = vid.read()
ret, frame = vid.read()
f2 = frame

gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
feature1 = cv2.cornerHarris(gray1, 2, 3, 0.04)

gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

p0 = []
thres = 0.1*feature1.max()
for i in range(480):
    for j in range(720):
        if feature1[i][j] > thres:
            p0.append([i, j])
p0 = np.float32((np.array(p0)).reshape(-1, 1, 2))

lk_params = dict(winSize=(15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1, str, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
p0r, st, err = cv2.calcOpticalFlowPyrLK(gray2, gray1, p1, None, **lk_params)

d = abs(p0-p0r).reshape(-1, 2).max(-1)

n = 0;
p0_new = []
p1_new = []
img = np.hstack((f1, f2))
for i in range(d.size):
    if d[i] < 0.005:
        n += 1;
        p0_new.append([int(p0[i][0][0]), int(p0[i][0][1])])
        p1_new.append([int(p1[i][0][0]), int(p1[i][0][1])])
        cv2.circle(img, (int(p0[i][0][1]), int(p0[i][0][0])), 5, (255, 0 ,0), 1)
        cv2.circle(img, (int(p1[i][0][1] + 720), int(p1[i][0][0])), 5, (255, 0, 0), 1)
        cv2.line(img, (int(p0[i][0][1]), int(p0[i][0][0])), (int(p1[i][0][1] + 720), int(p1[i][0][0])),(255,255,0),thickness=2)
plt.figure(1)
plt.imshow(img), plt.title('Matched Features')

x = np.zeros((2*n, 1))
A = np.zeros((2*n, 8))
for i in range(n):
    x[i] = p1_new[i][1]
    A[i][0] = 1
    A[i][1] = p0_new[i][1]
    A[i][2] = p0_new[i][0]
    A[i][3] = 0
    A[i][4] = 0
    A[i][5] = 0
    A[i][6] = -p0_new[i][1]*p1_new[i][1]
    A[i][7] = -p0_new[i][0]*p1_new[i][1]
for j in range(n):
    x[j+n] = p1_new[j][0]
    A[j+n][0] = 0
    A[j+n][1] = 0
    A[j+n][2] = 0
    A[j+n][3] = 1
    A[j+n][4] = p0_new[j][1]
    A[j+n][5] = p0_new[j][0]
    A[j+n][6] = -p0_new[j][1]*p1_new[j][0]
    A[j+n][7] = -p0_new[j][0]*p1_new[j][0]

a = np.dot(A.T, np.dot(np.linalg.inv(np.dot(A, A.T)), x))

for i in range(2000):
    a = a + 1e-13*np.dot(A.T, x - np.dot(A, a))
M = np.array([[a[1], a[2], a[0]],
              [a[4], a[5], a[3]],
              [a[6], a[7], 1]],
             dtype=np.float32)
print(M)

f1_new = cv2. warpPerspective(f1, M, (720, 480))
plt.figure(2)
plt.subplot(3, 1, 1), plt.imshow(f1), plt.title('1st Frame')
plt.subplot(3, 1, 2), plt.imshow(f2), plt.title('4th Frame')
plt.subplot(3, 1, 3), plt.imshow(f1_new), plt.title('4th Frame after homography transform')

fgmask = fgbg.apply(f1_new)
fgmask = fgbg.apply(f2)

plt.figure(3)
plt.imshow(fgmask, cmap=plt.cm.gray)
plt.show()
