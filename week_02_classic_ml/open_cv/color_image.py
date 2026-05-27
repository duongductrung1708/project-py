import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh mặc định (BGR)
img_bgr = cv2.imread("./resource/cat.jpg", cv2.IMREAD_COLOR)

if img_bgr is not None:
    # 2. Chuyển đổi BGR -> Grayscale (Tensor 3D -> 2D)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"Shape ảnh gốc BGR: {img_bgr.shape}")
    print(f"Shape ảnh Grayscale: {img_gray.shape}")

    # 3. Fix lỗi Matplotlib (BGR -> RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4. Tách các kênh màu bằng NumPy (Nhanh hơn cv2.split)
    # Cú pháp: [Tất cả hàng, Tất cả cột, Index của Kênh]
    b_channel = img_bgr[:, :, 0]
    g_channel = img_bgr[:, :, 1]
    r_channel = img_bgr[:, :, 2]
    print(f"Shape của 1 kênh màu (ví dụ Blue): {b_channel.shape} -> Đây là ảnh 2D đen trắng!")

    # 5. [Trực quan hóa Thực tế] 
    # Để màn hình hiện ra "Màu Xanh", ta phải tạo Tensor 3D với Red=0, Green=0
    # Khởi tạo ma trận toàn số 0 có cùng shape với ảnh gốc
    blue_visual = np.zeros_like(img_bgr)
    
    # Bơm ma trận kênh Blue (2D) vào không gian Kênh 0 của ma trận 3D
    blue_visual[:, :, 0] = b_channel
    
    # Hiển thị
    cv2.imshow("Original BGR", img_bgr)
    cv2.imshow("Grayscale", img_gray)
    cv2.imshow("Blue Channel (Visualized in 3D)", blue_visual)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không load được ảnh, check lại path!")