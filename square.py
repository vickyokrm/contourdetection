import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image = cv2.imread('tech_diagram1.png')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_purple = np.array([145, 100, 100])
upper_purple = np.array([160, 255, 255])

mask = cv2.inRange(hsv, lower_purple, upper_purple)

kernel = np.ones((3, 3), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
mask_sharp = cv2.filter2D(mask_clean, -1, sharpen_kernel)

contours, _ = cv2.findContours(mask_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
box_count = 0

for i, cnt in enumerate(contours):
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4 and cv2.contourArea(cnt) > 500:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h != 0 else 0

        if 0.9 <= aspect_ratio <= 1.1:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 195, 0), 2)
            roi = image[y:y+h, x:x+w]
            #cv2.imwrite(f'box_{i}.png', roi)
            box_count += 1

print(f"Total purple **square** boxes detected: {box_count}")
cv2_imshow(output)
