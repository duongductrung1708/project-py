import cv2
import numpy as np

# 1. Đọc ảnh
image = cv2.imread("./resource/cat.jpg")

if image is None:
    print("LỖI: Không tìm thấy ảnh!")
    exit()

# 2. Định nghĩa vùng muốn tô đen (Hình vuông 100x100 ở góc trên cùng bên trái)
# Cú pháp NumPy: [y_start:y_end, x_start:x_end]
# y từ 0 đến 100, x từ 0 đến 100
image[0:100, 0:100] = [0, 0, 0] 

# 3. Hiển thị kết quả
cv2.imshow('Anh da bi to den goc', image)
cv2.waitKey(0)
cv2.destroyAllWindows()