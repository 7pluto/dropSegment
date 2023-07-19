import os

import cv2
from matplotlib import pyplot as plt
from datetime import datetime
import cv2
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN, KMeans

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def remove_illumination(gray):
    gray = cv2.convertScaleAbs(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized


def retinex_multiscale(gray, sigma_list):
    result = np.zeros_like(gray, dtype=np.float32)
    for sigma in sigma_list:
        blur_image = cv2.GaussianBlur(gray, (0, 0), sigma)
        reflectance = np.log1p(gray) - np.log1p(blur_image)
        result += reflectance

    result /= len(sigma_list)
    result = np.expm1(result)
    result = (result * 255).astype(np.uint8)
    return result


for i in range(1, 50):
    for j in range(10):
        path = "C:\\Users\\chenwh\\PycharmProjects\\dropSeg\\outputs\\" + str(i)
        image_path = path + "\\img_single__ROI_" + str(j) + ".jpg"
        print(image_path)

        image = cv2.imread(image_path)

        # gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        image_equalized = remove_illumination(gray)
        ret2, th2 = cv2.threshold(image_equalized, 0, 255, cv2.THRESH_OTSU)

        image_float = image.astype(np.float32) / 255.0
        sigma_list = [15, 80, 250]
        enhanced_image = retinex_multiscale(image_float, sigma_list)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        ret3, th3 = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_OTSU)

        # RGB
        B, G, R = cv2.split(image)
        mask = (R > G) & (R > B)
        R = R * mask
        G = G * mask
        B = B * mask
        new_img = cv2.merge([B, G, R])
        new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # _, _, new_gray = cv2.split(image)
        new_ret, new_th = cv2.threshold(new_gray, 0, 255, cv2.THRESH_OTSU)

        image_equalized_2 = remove_illumination(new_gray)
        ret5, th5 = cv2.threshold(image_equalized_2, 0, 255, cv2.THRESH_OTSU)

        image_float = new_img.astype(np.float32) / 255.0
        sigma_list = [15, 80, 250]
        enhanced_image_2 = retinex_multiscale(image_float, sigma_list)
        _, _, enhanced_image_2 = cv2.split(enhanced_image_2)
        ret6, th6 = cv2.threshold(enhanced_image_2, 0, 255, cv2.THRESH_OTSU)

        gray_image = np.float32(new_gray)
        kmeans_clusters = 2  # K-means聚类数目
        kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10)
        kmeans.fit(gray_image.reshape(-1, 1))
        segmented_image_kmeans = kmeans.labels_.reshape(gray_image.shape)
        binary_image_kmeans = np.zeros_like(segmented_image_kmeans, dtype=np.uint8)
        binary_image_kmeans[segmented_image_kmeans == np.argmax(np.bincount(segmented_image_kmeans.flatten()))] = 255

        intersection_image = np.logical_and(new_th, th).astype(np.uint8) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(241), plt.imshow(image, "gray")
        plt.title("source"), plt.xticks([]), plt.yticks([])

        plt.subplot(242), plt.imshow(th, "gray")
        plt.title("gray " + str(ret)), plt.xticks([]), plt.yticks([])

        plt.subplot(243), plt.imshow(new_th, "gray")
        plt.title("RGB split " + str(new_ret)), plt.xticks([]), plt.yticks([])

        plt.subplot(244), plt.imshow(binary_image_kmeans, "gray")
        plt.title("k-means"), plt.xticks([]), plt.yticks([])

        plt.subplot(245), plt.imshow(th3, "gray")
        plt.title("hgray" + str(ret3)), plt.xticks([]), plt.yticks([])

        plt.subplot(246), plt.imshow(th2, "gray")
        plt.title("qgray" + str(ret2)), plt.xticks([]), plt.yticks([])

        plt.subplot(247), plt.imshow(th6, "gray")
        plt.title("hRGB" + str(ret6)), plt.xticks([]), plt.yticks([])

        plt.subplot(248), plt.imshow(th5, "gray")
        plt.title("qRGB" + str(ret5)), plt.xticks([]), plt.yticks([])

        # plt.show()

        new_folder_path = "C:\\Users\\chenwh\\PycharmProjects\\dropSeg\\illumination_outputs\\" + str(i)
        os.makedirs(new_folder_path, exist_ok=True)
        filename = f"{new_folder_path}\\leaf_{i}_{j}.png"
        print(filename)
        plt.savefig(filename)

