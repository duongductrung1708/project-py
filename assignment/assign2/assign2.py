import cv2
import numpy as np

INPUT_PATH = "../data/Croissant.jpg"
OUTPUT_PATH = "./output/croissant_cropped.png"

PADDING = 20

image = cv2.imread(INPUT_PATH)

if image is None:
    raise ValueError("Không đọc được ảnh")

# =====================================
# Detect artwork
# =====================================

background_color = image[0, 0].astype(np.int16)

diff = np.abs(
    image.astype(np.int16) - background_color
)

distance = np.max(diff, axis=2)

mask = np.where(
    distance > 15,
    255,
    0
).astype(np.uint8)

# =====================================
# Morphology
# =====================================

kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# =====================================
# Bounding box
# =====================================

points = cv2.findNonZero(mask)

if points is None:
    raise ValueError("Không tìm thấy artwork")

x, y, w, h = cv2.boundingRect(points)

# =====================================
# Padding
# =====================================

x1 = max(0, x - PADDING)
y1 = max(0, y - PADDING)

x2 = min(image.shape[1], x + w + PADDING)
y2 = min(image.shape[0], y + h + PADDING)

# =====================================
# Crop
# =====================================

cropped = image[y1:y2, x1:x2]

# =====================================
# Save
# =====================================

cv2.imwrite(OUTPUT_PATH, cropped)

print("=" * 50)
print("AUTO CROP COMPLETED")
print("=" * 50)
print(f"x={x}")
print(f"y={y}")
print(f"w={w}")
print(f"h={h}")
print()
print(f"Crop size: {cropped.shape}")
print(f"Saved: {OUTPUT_PATH}")

# Debug

bbox_debug = image.copy()

cv2.rectangle(
    bbox_debug,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.imwrite("mask.png", mask)
cv2.imwrite("bbox_debug.png", bbox_debug)

cv2.imshow("Mask", mask)
cv2.imshow("Crop", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()