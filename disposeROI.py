import cv2
from matplotlib import pyplot as plt
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

image = cv2.imread("./output/ot_1320606.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

B, G, R = cv2.split(image)
mask = (R > G) & (R > B)
R = R * mask
G = G * mask
B = B * mask
new_img = cv2.merge([B, G, R])

new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_ret1, new_th1 = cv2.threshold(new_gray, 0, 255, cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
print("通道分离:", new_ret1)
print(ret2)


new_contours, new_hierarchy = cv2.findContours(new_th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
new_num_contours = len(new_contours)
new_areas = [cv2.contourArea(contour) for contour in new_contours]
# cv2.drawContours(image, contours, -1, (255, 0, 0), 20)

contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_contours = len(contours)
areas = [cv2.contourArea(contour) for contour in contours]
# cv2.drawContours(image, contours, -1, (255, 0, 0), 20)

print("通道分离轮廓个数:", new_num_contours)
print("通道分离每个轮廓的面积:", new_areas)
print("通道分离轮廓总面积:", sum(new_areas))
print("轮廓个数:", num_contours)
print("每个轮廓的面积:", areas)
print("轮廓总面积:", sum(areas))


cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("new_gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("new th1", cv2.WINDOW_NORMAL)
cv2.namedWindow("th2", cv2.WINDOW_NORMAL)
cv2.imshow("img", image)
cv2.imshow("gray", gray)
cv2.imshow("new_gray", new_gray)
cv2.imshow("new th1", new_th1)
cv2.imshow("th2", th2)

cv2.waitKey(0)

# plt.subplot(121), plt.imshow(image, "gray")
# plt.title("source image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(th1, "gray")
# plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
# # plt.show()
#
# filename = f"plt/leaf_{ret1}_{current_time}.png"
# plt.savefig(filename)


