import os
import largestinteriorrectangle as lir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def maximum_internal_rectangle_2(img, contours):
    contour = contours[0].reshape(len(contours[0]), 2)
    rect = []
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))

    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)

        while not best_rect_found and index_rect < nb_rect:

            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]

            valid_rect = True

            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1

            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1

            if valid_rect:
                best_rect_found = True

            index_rect += 1

        if best_rect_found:
            # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # 在mask上画最小内接矩形
            # cv2.imshow("rec", img)
            return img, x1, y1, x2, y2
            # cv2.waitKey(0)

        else:
            print("No rectangle fitting into the area")

    else:
        print("No rectangle found")


def random_crop_inner_rectangle(img, x1, y1, x2, y2, crop_width, crop_height):
    # 确保x1 < x2 和 y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # 计算裁剪的范围，考虑crop_width和crop_height的一半
    max_x = x2 - crop_width // 2
    min_x = x1 + crop_width // 2
    max_y = y2 - crop_height // 2
    min_y = y1 + crop_height // 2

    # 创建一个空白图像作为默认返回值
    cropped_img = np.zeros((crop_height, crop_width), np.uint8)

    # 检查裁剪范围是否有效，有效则随机裁剪
    if min_x < max_x and min_y < max_y:
        crop_x = np.random.randint(min_x, max_x)
        crop_y = np.random.randint(min_y, max_y)

        # 裁剪图像
        cropped_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    return cropped_img


model_type = "vit_t"
sam_checkpoint = "C:\\Users\\chenwh\\Downloads\\MobileSAM-master\\weights\\mobile_sam.pt"
device = "cuda"
image_name = "img_single_"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)
# 图像文件名
# img_single_1.jpg

for i in range(1, 50):
    image_path = "E:/data_images/" + image_name + str(i) + ".jpg"
    image = cv2.imread(image_path)

    predictor.set_image(image)
    input_point = np.array([[1600, 1800]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(input_point,
        point_labels=input_label,
        multimask_output=False,)

    mask = ~masks[0] + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    res = cv2.bitwise_and(image, mask)

    # 为加速计算，将image进行resize后再求最小内接矩形
    img = mask
    scale_percent = 9
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # 求最小内接矩形
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    src, x1, y1, x2, y2 = maximum_internal_rectangle_2(resized, contours)

    # 将矩形放大回原来的大小
    x1 = int(x1 / scale_percent * 100)
    x2 = int(x2 / scale_percent * 100)
    y1 = int(y1 / scale_percent * 100)
    y2 = int(y2 / scale_percent * 100)

    # 查看提示点与叶片的位置关系
    # for k, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.suptitle(image_name + str(i), fontsize=20)
    #     plt.imshow(image)
    #     show_mask(masks[k], plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {k}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #
    #     plt_name = "plt/" + image_name + str(i) + "_" + str(k) +".png"
    #     plt.savefig(plt_name)
    #     plt.show()

    # 在原图和mask上画最小内接矩形
    image_rect = image.copy()
    cv2.rectangle(image_rect, (x1, y1), (x2, y2), (255, 0, 0), 20)
    # cv2.namedWindow("image_rect", cv2.WINDOW_NORMAL)
    # cv2.imshow("image_rect", image_rect)
    # cv2.imshow("src", src)
    # cv2.waitKey(0)

    new_folder_path = "rect_outputs/" + str(i)
    os.makedirs(new_folder_path, exist_ok=True)

    # 保存叶片黑色背景图
    filename2 = f"{new_folder_path}/{image_name}_leaf.jpg"
    cv2.imwrite(filename2, res)
    # 保存原图
    filename3 = f"{new_folder_path}/{image_name}_image.jpg"
    cv2.imwrite(filename3, image)
    # 原图和mask上画最小内接矩形
    filename4 = f"{new_folder_path}/{image_name}_image_rect.jpg"
    cv2.imwrite(filename4, image_rect)
    filename5 = f"{new_folder_path}/{image_name}_mask_rect.jpg"
    cv2.imwrite(filename5, src)

    for j in range(3):
        crop_width = (x2 - x1) * 0.3
        crop_height = (y2 - y1) * 0.3

        random_rectangle = random_crop_inner_rectangle(image, x1, y1, x2, y2, 300, 300)

        # random_rectangle = cv2.cvtColor(random_rectangle, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, j + 1)
        plt.imshow(random_rectangle)
        plt.xticks([]), plt.yticks([])

        # cv2.imshow("random_rectangle", random_rectangle)
        # cv2.waitKey(0)

        filename = f"{new_folder_path}/{image_name}_ROI_{j}.jpg"
        cv2.imwrite(filename, random_rectangle)
    #
    # plt.show()
    # cv2.waitKey(0)

    print("已处理： " + image_path)


