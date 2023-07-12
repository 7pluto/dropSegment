import cv2
from matplotlib import pyplot as plt
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

image = cv2.imread("./images/drop_0.jpg")
#image = cv2.imread("./output/ot_4310900.jpg")

B,G,R = cv2.split(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = cv2.subtract(R, G)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", image)

cv2.namedWindow("gray image", cv2.WINDOW_NORMAL)
cv2.imshow("gray image", gray)


_, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.namedWindow("threshold_image", cv2.WINDOW_NORMAL)
cv2.imshow("threshold_image", threshold_image)


contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (255, 0, 0), 20)


#ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
ret1, th1 = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

fig, ax = plt.subplots()
plt.subplot(121), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(th1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
#plt.show()

filename = f"plt/leaf_{ret1}_{current_time}.png"
plt.savefig(filename)


