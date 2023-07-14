import cv2
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


image = cv2.imread('./output/ot_1320606.jpg')

B,G,R = cv2.split(image)
mask = (R > G) & (R > B)
R = R * mask
G = G * mask
B = B * mask
new_img = cv2.merge([B, G, R])

gray_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("Meanshift", cv2.WINDOW_NORMAL)
cv2.namedWindow("K-means", cv2.WINDOW_NORMAL)
cv2.imshow('Image', image)
cv2.imshow("gray", gray_image)

# 转换为浮点型数据
gray_image = np.float32(gray_image)

# 定义聚类算法的参数
meanshift_bandwidth = 27  # Meanshift带宽参数
dbscan_eps = 6  # DBSCAN的邻域半径参数
dbscan_min_samples = 200  # DBSCAN的最小样本数参数
kmeans_clusters = 2  # K-means聚类数目
linkage_method = 'ward'  # 层次聚类的链接方法


# 运行Meanshift聚类算法
# meanshift = MeanShift(bandwidth=meanshift_bandwidth)
# meanshift.fit(gray_image.reshape(-1, 1))
# segmented_image_meanshift = meanshift.labels_.reshape(gray_image.shape)
# binary_image_meanshift = np.zeros_like(segmented_image_meanshift, dtype=np.uint8)
# binary_image_meanshift[segmented_image_meanshift == np.argmax(np.bincount(segmented_image_meanshift.flatten()))] = 255
# cv2.imshow('Meanshift', binary_image_meanshift)


# 运行DBSCAN聚类算法
# dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
# dbscan.fit(gray_image.reshape(-1, 1))
# segmented_image_dbscan = dbscan.labels_.reshape(gray_image.shape)
# binary_image_dbscan = np.zeros_like(segmented_image_dbscan, dtype=np.uint8)
# binary_image_dbscan[segmented_image_dbscan != -1] = 255
# cv2.imshow('Binary Image (DBSCAN)', binary_image_dbscan)
#

# 运行K-means聚类算法
kmeans = KMeans(n_clusters=kmeans_clusters)
kmeans.fit(gray_image.reshape(-1, 1))
segmented_image_kmeans = kmeans.labels_.reshape(gray_image.shape)
binary_image_kmeans = np.zeros_like(segmented_image_kmeans, dtype=np.uint8)
binary_image_kmeans[segmented_image_kmeans == np.argmax(np.bincount(segmented_image_kmeans.flatten()))] = 255
cv2.imshow('K-means', binary_image_kmeans)


# 运行层次聚类算法
# from scipy.spatial.distance import pdist
# distances = pdist(gray_image.flatten().reshape(-1, 1))
# linked = linkage(distances, method=linkage_method)
# segmented_image_hierarchy = np.zeros_like(gray_image, dtype=np.uint8)
# dendrogram(linked, no_plot=True, color_threshold=0, distance_sort=True)
# clusters = np.unique(linked[:, 3])
# for i, cluster in enumerate(clusters):
#     indices = np.where(linked[:, 3] == cluster)[0]
#     segmented_image_hierarchy[np.isin(segmented_image_hierarchy, indices)] = i
# binary_image_hierarchy = np.zeros_like(segmented_image_hierarchy, dtype=np.uint8)
# binary_image_hierarchy[segmented_image_hierarchy == np.argmax(np.bincount(segmented_image_hierarchy.flatten()))] = 255
# cv2.imshow('Binary Image (Hierarchy)', binary_image_hierarchy)



cv2.waitKey(0)
cv2.destroyAllWindows()
