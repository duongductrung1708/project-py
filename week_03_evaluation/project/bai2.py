import cv2
import numpy as np

image = cv2.imread("./data/bai2/4a47b89c0cd6457c8b7858c071b69071.png", cv2.IMREAD_UNCHANGED)

alpha = image[:, :, 3]

blurred = cv2.GaussianBlur(alpha, (3, 3), 0)

_, binary = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = np.ones((3,3), np.uint8)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours found:", len(contours))

total_objects = 0

vertical_count = 0

horizontal_count = 0

circle_count = 0

image_vis = cv2.cvtColor(
    binary,
    cv2.COLOR_GRAY2BGR
)

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 50:
        continue
    total_objects += 1
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity > 0.85 and 0.8 < aspect_ratio < 1.2: 
        circle_count += 1
        color = (0,255,0)
        label = "Circle"
    elif aspect_ratio < 0.8:
        vertical_count += 1
        color = (255,0,0)
        label = "Vertical"
    else:
        horizontal_count += 1
        color = (0,0,255)
        label = "Horizontal"
    
    cv2.drawContours(image_vis, [contour], -1, color, 2)

print("Total Objects :", total_objects)
print("Vertical      :", vertical_count)
print("Horizontal    :", horizontal_count)
print("Circle        :", circle_count)

cv2.imshow("Result", image_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()