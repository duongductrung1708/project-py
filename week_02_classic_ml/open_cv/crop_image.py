import cv2
import os

# 1. Đọc ảnh
img = cv2.imread('./resource/cat.jpg')
if img is None:
    print("Không tìm thấy file cat.jpg!")
    exit()

# 2. Lấy kích thước ảnh gốc
height, width, _ = img.shape

# 3. Tính toán vùng cắt (Trung tâm của ảnh)
# Chúng ta lấy 60% chiều rộng và 70% chiều cao ở giữa ảnh
crop_w = int(width * 0.6)
crop_h = int(height * 0.7)

# Tính toán điểm bắt đầu (x, y) để cắt đúng tâm
start_x = (width - crop_w) // 2
start_y = (height - crop_h) // 2

# Cắt ảnh
cropped_image = img[start_y : start_y + crop_h, start_x : start_x + crop_w]

# 4. Hiển thị để bạn kiểm tra vị trí
cv2.namedWindow("Face Crop Preview", cv2.WINDOW_NORMAL)
cv2.imshow("Face Crop Preview", cropped_image)

# Lưu lại
cv2.imwrite("Face_Cat_Corrected.jpg", cropped_image)
print(f"Ảnh đã cắt: {crop_h}x{crop_w}. Vị trí bắt đầu: x={start_x}, y={start_y}")

cv2.waitKey(0)
cv2.destroyAllWindows()
