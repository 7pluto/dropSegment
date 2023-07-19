import os

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


model_type = "vit_t"
sam_checkpoint = "C:\\Users\\chenwh\\Downloads\\MobileSAM-master\\weights\\mobile_sam.pt"
device = "cuda"
image_name = "img_single_"

for i in range(1, 50):
    image_path = "E:/data_images/" + image_name + str(i) + ".jpg"
    image = cv2.imread(image_path)


    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
    predictor.set_image(image)
    input_point = np.array([[1800, 1700]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(input_point,
        point_labels=input_label,
        multimask_output=False,)


    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(masks[i], plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()

    new_folder_path = "outputs/" + str(i)
    os.makedirs(new_folder_path, exist_ok=True)

    mask = masks[0].astype(np.uint8)
    mask = ~masks[0]
    mask = mask + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    res = cv2.bitwise_and(image, mask)

    filename2 = f"{new_folder_path}/{image_name}_leaf.jpg"
    cv2.imwrite(filename2, res)

    filename3 = f"{new_folder_path}/{image_name}_image.jpg"
    cv2.imwrite(filename3, image)

    for j in range(10):
        indices = np.where(masks[0] != 0)
        random_index = np.random.choice(len(indices[0]))
        random_x, random_y = indices[1][random_index], indices[0][random_index]
        rect_width, rect_height = 300, 300
        rect_x = max(random_x - rect_width // 2, 0)
        rect_y = max(random_y - rect_height // 2, 0)
        random_rectangle = image[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]


        filename = f"{new_folder_path}/{image_name}_ROI_{j}.jpg"
        cv2.imwrite(filename, random_rectangle)


    print("已处理： " + image_path)


