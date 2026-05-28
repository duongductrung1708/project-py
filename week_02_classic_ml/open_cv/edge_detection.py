import cv2
import numpy as np

# 1. Đọc ảnh và kiểm tra tồn tại
img = cv2.imread('./resource/cat.jpg')
if img is None:
    print("Không tìm thấy ảnh './resource/cat.jpg'")
    exit()

# Resize về chiều ngang 800px, giữ nguyên tỷ lệ
h, w = img.shape[:2]
new_w = 800
new_h = int(h * (new_w / w))
img_resized = cv2.resize(img, (new_w, new_h))
print(f"Resize: {w}x{h} → {new_w}x{new_h}")

# 2. Chuyển xám và làm mờ
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow('Gray', gray)
cv2.imshow('Blur', img_blur)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
 
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
 
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
 
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# 3. Auto-Canny
v = np.median(img_blur)
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
edges = cv2.Canny(img_blur, lower, upper)

# Hiển thị
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()