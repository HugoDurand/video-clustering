import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from skimage import io
from skimage import img_as_float64
from sklearn.decomposition import PCA
import cv2


def reconstruct_image(cluster_centers, labels, w, h):
    d = cluster_centers.shape[1]
    image = np.zeros((w, h, d))
    label_index = 0
    for i in range (w):
        for j in range(h):
            image[i][j] = cluster_centers[labels[label_index]]
            label_index += 1
    return image

def convert_3d_matrix_to_2d(matrix):
    w, h, d = tuple(matrix.shape)
    assert d == 3
    twoD_matrix = np.reshape(converted_image, (w*h, d))
    return twoD_matrix

# ============================
# EXTRACT FRAMES FROM VIDEO
# ============================

# video = cv2.VideoCapture('./videos/input.mp4')
# i = 0
# frameRate = video.get(5)
# while video.isOpened():
#     frameId = video.get(1)
#     ret, frame = video.read()
#     if ret == False:
#         break
#     if frameId % math.floor(frameRate) == 0:
#         cv2.imwrite('images/img'+str(i)+'.jpg', frame)
#     i+=1
# video.release()
# cv2.destroyAllWindows()

# ============================
# COLOR QUANTIZATION
# ============================

# === LOAD IMAGE ===
imageCollection = io.MultiImage('./images/*.jpg')

tmp_array = []

# === CONVERT 8 BIT TO FLOAT ===
for image in imageCollection:
    converted_image = img_as_float64(image)

    # === CONVERT IMAGE INTO 2D MATRIX FOR MANIPULATION ===
    image_array = convert_3d_matrix_to_2d(converted_image)

    # === TRAIN MODEL TO AGGREGATE COLORS IN ORDER TO HAVE 64 DISTINCT COLORS IN IMAGE ===
    image_sample = shuffle(image_array, random_state=0)[:1000]
    n_colors = 64
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_sample)
    labels = kmeans.predict(image_array)

    plt.imshow(reconstruct_image(kmeans.cluster_centers_, labels, w, h))
    plt.show()
    tmp_array.append(reconstruct_image(kmeans.cluster_centers_, labels, w, h))


# images = np.stack(tmp_array, axis=0)
# for img in images:
#     plt.imshow(img)
#     plt.show()
