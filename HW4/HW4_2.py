# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

# Load the Summer Palace photo
china = load_sample_image("china.jpg")
china1 = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
# kmeans = KMeans(n_clusters=n_colors, random_state=5, n_init=1, verbose=1).fit(image_array_sample)
kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10,).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,image_array,axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    # d = codebook.shape[1]
    d = 3
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
# plt.figure(1)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)
#
# plt.figure(2)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
#
# plt.figure(3)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(recreate_image(codebook_random, labels_random, w, h))
# plt.show()
#
# plt.figure(4)
# x2 = [0,1,2,3,4,5,6,7,8,9,10]
# y2 = [1.982,1.553,1.485,1.466,1.461,1.459,1.457,1.454,1.454,1.453,1.453]
#
# plt.title('error reduction curve(K-Means++, 1 trial)')
# plt.xlabel('iteration time')
# plt.ylabel('error')
# plt.plot(x2,y2)
# plt.show()
#
#
# plt.figure(5)
# x1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# y1 = [4.094,2.491
#    ,2.130
#    ,2.047
#    ,2.015
#    ,1.998
#    ,1.989
#    ,1.971
#    ,1.964
#    ,1.961
#    ,1.959
#    ,1.958
#    ,1.957
#    ,1.956
#    ,1.955]
#
# plt.title('error reduction curve(random selection, 1 trial)')
# plt.xlabel('iteration time')
# plt.ylabel('error')
# plt.plot(x1,y1)
# plt.show()


x = np.array([image_array[1,:],0,0])

addone = np.ones((1,2))
result1 = np.hstack((china[1,1,:],addone[0,:]))

# print image_array
print 'image_array.shape:',image_array.shape
print 'shape of addone:',addone.shape
print 'shape of china:',china.shape


new = np.array(np.zeros((w,h,5)))
new[0,0,:] = np.hstack((china[1,1,:],addone[0,:]))


coordinate_mat_x = np.array(np.zeros((w,h,2)))
coordinate_mat_y = np.array(np.zeros((w,h)))

for i in range(1,w):
    for j in range(1,h):
        coordinate_mat_x[i,j,:] = [i,j]

# print coordinate_mat_x[1,:,:]
# print coordinate_mat_x[:,1,:]


new_imga_mat = np.array(np.zeros((w,h,5)))
new_imga_mat[0,0,:] = np.hstack((china[0,0,:],coordinate_mat_x[0,0,:]))
# print 'new_imga_mat:',new_imga_mat[0,0,:]
# print 'china:', china[0,0,:]
# print 'coordiante:',coordinate_mat_x[0,0,:]
# print china

weight = 1

revised_coordinate = coordinate_mat_x
revised_coordinate[:,:,0] = coordinate_mat_x[:,:,0]/w*weight
revised_coordinate[:,:,1] = coordinate_mat_x[:,:,1]/h*weight

# print 'last_value:',revised_coordinate[w-1,h-1,:]
# print  revised_coordinate
# print revised_coordinate
# print revised_coordinate.shape
# print 'china:',china
# print 'revised_coordiante',coordinate_mat_x/w*255*1
# print 'coordinate:', coordinate_mat_x



china_revised = np.array(np.zeros((w,h,5)))
for a in range(1,w):
    for b in range(1,h):
        china_revised[a,b,:] = np.hstack((china[a,b,:],revised_coordinate[a,b,:]))

print 'china:',china[5,5]
print 'revised_china:',china_revised[5,5]
print '\n'


# Load Image and transform to a 2D numpy array.
w1, h1, d1 = original_shape1 = tuple(china_revised.shape)
assert d1 == 5
image_array1 = np.reshape(china_revised, (w1 * h1, d1))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample1 = shuffle(image_array1, random_state=0)[:1000]
# kmeans = KMeans(n_clusters=n_colors, random_state=5, n_init=1, verbose=1).fit(image_array_sample)
kmeans1 = KMeans(n_clusters=n_colors, random_state=0,n_init=10).fit(image_array_sample1)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels1 = kmeans1.predict(image_array1)
print("done in %0.3fs." % (time() - t0))


codebook_random1 = shuffle(image_array1, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random1 = pairwise_distances_argmin(codebook_random1,image_array1,axis=0)
print("done in %0.3fs." % (time() - t0))

print image_array_sample.shape
print image_array_sample1.shape
print image_array_sample[1,:]
print image_array_sample1[1,:]

print labels[1:10]
print labels1[1:10]
print recreate_image(kmeans.cluster_centers_, labels1, w1, h1)
print '\n'
print recreate_image(kmeans.cluster_centers_, labels, w, h)


# plt.figure(1)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels1, w1, h1))


plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random1, w1, h1))
plt.show()

print china[400,600,:]
print china_revised[400,600,:]
print china1[400,600,:]