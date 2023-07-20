import os
import cv2
from matplotlib import pyplot as plt
from datetime import datetime
import cv2
import numpy as np
from sklearn.cluster import KMeans

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def Clahe(gray_image):
    gray_image = cv2.convertScaleAbs(gray_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_image)
    return img_clahe


def retinex_multiscale(gray_image, sigma_list):
    result = np.zeros_like(gray_image, dtype=np.float32)
    for sigma in sigma_list:
        blur_image = cv2.GaussianBlur(gray_image, (0, 0), sigma)
        reflectance = np.log1p(gray_image) - np.log1p(blur_image)
        result += reflectance

    result /= len(sigma_list)
    result = np.expm1(result)
    img_retinex = (result * 255).astype(np.uint8)
    return img_retinex


def extract_filename(image_path):
    # Split the image path using the directory separator '/'
    filename = image_path.split(os.path.sep)[-1]
    return filename


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def plt_show(n, m, plt_list):
    show_x = 3
    show_y = int(len(plt_list)/3+1)
    for i in range(len(plt_list)):
        image = cv2.cvtColor(plt_list[i], cv2.COLOR_BGR2RGB)
        plt.suptitle("img_single__ROI_" + str(n) + "_" + str(m), fontsize=15)
        plt.subplot(show_y, show_x, i+1)
        plt.imshow(image)
        plt.title(namestr(plt_list[i], globals())[0])
        plt.xticks([]), plt.yticks([])

    new_folder_path = "C:\\Users\\chenwh\\PycharmProjects\\dropSeg\\rect_outputs\\plt"
    os.makedirs(new_folder_path, exist_ok=True)
    filename = f"{new_folder_path}\\leaf_{n}_{m}.png"
    print(filename)
    plt.show()


show_list = []
for i in range(1, 50):
    for j in range(3):
        path = "C:\\Users\\chenwh\\PycharmProjects\\dropSeg\\rect_outputs\\" + str(i)
        image_path = path + "\\img_single__ROI_" + str(j) + ".jpg"
        print(image_path)

        image = cv2.imread(image_path)

        # Gray
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, threshold_gray = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

        image_equalized = Clahe(gray_image)
        ret2, img_clahe = cv2.threshold(image_equalized, 0, 255, cv2.THRESH_OTSU)

        image_float = image.astype(np.float32) / 255.0
        sigma_list = [15, 80, 250]
        enhanced_image = retinex_multiscale(image_float, sigma_list)
        enhanced_gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        ret3, img_retinex = cv2.threshold(enhanced_gray_image, 0, 255, cv2.THRESH_OTSU)

        # RGB
        B, G, R = cv2.split(image)
        mask = (R > G) & (R > B)
        R = R * mask
        G = G * mask
        B = B * mask
        new_img = cv2.merge([B, G, R])
        new_gray_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        ret4, threshold_rgb = cv2.threshold(new_gray_image, 0, 255, cv2.THRESH_OTSU)

        image_equalized_2 = Clahe(new_gray_image)
        ret5, img_clahe_rgb = cv2.threshold(image_equalized_2, 0, 255, cv2.THRESH_OTSU)

        image_float = new_img.astype(np.float32) / 255.0
        sigma_list = [15, 80, 250]
        enhanced_image_2 = retinex_multiscale(image_float, sigma_list)
        _, _, enhanced_gray_image_2 = cv2.split(enhanced_image_2)
        ret6, img_retinex_rgb = cv2.threshold(enhanced_gray_image_2, 0, 255, cv2.THRESH_OTSU)

        gray_image_float = np.float32(new_gray_image)
        kmeans_clusters = 2  # K-means聚类数目
        kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10)
        kmeans.fit(gray_image_float.reshape(-1, 1))
        segmented_image_kmeans = kmeans.labels_.reshape(gray_image_float.shape)
        binary_image_kmeans = np.zeros_like(segmented_image_kmeans, dtype=np.uint8)
        binary_image_kmeans[segmented_image_kmeans == np.argmax(np.bincount(segmented_image_kmeans.flatten()))] = 255

        intersection_image = np.logical_and(threshold_rgb, threshold_gray).astype(np.uint8) * 255

        show_list = [image, threshold_gray, threshold_rgb, img_retinex, img_clahe, img_clahe_rgb, img_retinex_rgb]

        plt_show(i, j, show_list)
