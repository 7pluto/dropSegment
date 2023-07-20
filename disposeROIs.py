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
    num_images = len(plt_list)
    show_y = num_images // show_x
    if num_images % show_x != 0:
        show_y += 1

    for i, image in enumerate(plt_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.suptitle("img_single__ROI_" + str(n) + "_" + str(m), fontsize=15)
        plt.subplot(show_y, show_x, i+1)
        plt.imshow(image)
        plt.title(namestr(plt_list[i], globals())[0])
        plt.xticks([]), plt.yticks([])

    new_folder_path = "C:\\Users\\chenwh\\PycharmProjects\\dropSeg\\rect_outputs\\plt"
    os.makedirs(new_folder_path, exist_ok=True)
    filename = f"{new_folder_path}\\leaf_{n}_{m}.png"
    print(filename)
    plt.savefig(filename)


def fill_closing(binary_img, kernel_size=11):
    # # 定义一个闭运算的结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #
    # # 使用闭运算填充缺失的轮廓中心
    # filled_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    #
    # return filled_img
    filled_img = binary_img.copy()
    inverted_binary = cv2.bitwise_not(filled_img)
    contours, _ = cv2.findContours(inverted_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area <= 200:
            cv2.drawContours(filled_img, [contour], -1, (0, 0, 0), cv2.FILLED)
    return filled_img


def fill_bilinear_interpolation(image, radius=5):
    # 复制输入图像，以免修改原始图像
    filled_image = image.copy()

    # 使用高斯模糊来模糊图像
    blurred_image = cv2.GaussianBlur(image, (2 * radius + 1, 2 * radius + 1), 0)

    # 获取二值化图像，将非零像素设为1
    _, binary_img = cv2.threshold(blurred_image, 1, 1, cv2.THRESH_BINARY)

    # 使用双线性插值来填充缺失的像素
    filled_image = filled_image.astype(float)
    binary_img = binary_img.astype(float)
    for i in range(filled_image.shape[0]):
        for j in range(filled_image.shape[1]):
            if binary_img[i, j] == 0:
                total_weight = 0
                total_value = 0
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = i + dx, j + dy
                        if nx >= 0 and nx < filled_image.shape[0] and ny >= 0 and ny < filled_image.shape[1]:
                            weight = np.exp(-(dx * dx + dy * dy) / (2 * radius * radius))
                            total_weight += weight
                            total_value += weight * filled_image[nx, ny]
                filled_image[i, j] = total_value / total_weight

    filled_image = filled_image.astype(np.uint8)

    return filled_image


show_list = []
for i in range(1, 50):
    for j in range(3):
        # i = 9 11 16
        # j = 0
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

        # gray_image_float = np.float32(new_gray_image)
        # kmeans_clusters = 2  # K-means聚类数目
        # kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10)
        # kmeans.fit(gray_image_float.reshape(-1, 1))
        # segmented_image_kmeans = kmeans.labels_.reshape(gray_image_float.shape)
        # binary_image_kmeans = np.zeros_like(segmented_image_kmeans, dtype=np.uint8)
        # binary_image_kmeans[segmented_image_kmeans == np.argmax(np.bincount(segmented_image_kmeans.flatten()))] = 255

        intersection_image = np.logical_and(threshold_rgb, threshold_gray).astype(np.uint8) * 255

        retinex_rgb_close = fill_closing(img_retinex_rgb)
        retinex_rgb_inter = fill_bilinear_interpolation(img_retinex_rgb)

        show_list = [image, threshold_gray, threshold_rgb, img_retinex, img_clahe, img_clahe_rgb, img_retinex_rgb, retinex_rgb_close, retinex_rgb_inter]

        plt_show(i, j, show_list)
