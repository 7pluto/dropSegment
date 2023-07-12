
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from timeit import default_timer as timer
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")



def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# 读取图像
image = cv2.imread('images/drop_1.jpg')
sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

# 获取mask
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)
input_point = np.array([[2000, 1000]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
mask = masks[0].astype(np.uint8)

# 截取部分区域图像
indices = np.where(mask != 0)
# 随机选择一个索引
random_index = np.random.choice(len(indices[0]))
# 获取随机索引对应的坐标
random_x, random_y = indices[1][random_index], indices[0][random_index]
# 定义截取矩形的宽度和高度
rect_width, rect_height = 300, 300
# 计算截取矩形的左上角坐标
rect_x = max(random_x - rect_width // 2, 0)
rect_y = max(random_y - rect_height // 2, 0)
# 截取矩形区域
random_rectangle = image[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
#cv2.imwrite("output/ot_" + str(random_index) + ".jpg", random_rectangle)


# 原图和分割叶片
mask = ~masks[0]
mask = mask + 255
mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
mask = mask.astype(np.uint8)
res = cv2.bitwise_and(image, mask)
#res[res == 0] = 255 #背景为白色
#cv2.imwrite("./output/drop.jpg", res)
plt.subplot(121), plt.imshow(image, "gray")
show_points(input_point, input_label, plt.gca())
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res, "gray")
plt.title("seg image"), plt.xticks([]), plt.yticks([])
plt.show()

# 对截取的对象做OTSU
image = random_rectangle
B,G,R = cv2.split(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = cv2.subtract(R, G)

ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
#ret1, th1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

plt.subplot(121), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(th1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
plt.show()