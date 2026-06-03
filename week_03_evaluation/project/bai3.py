import cv2
import numpy as np

# 1. Đọc ảnh MÀU (dùng IMREAD_COLOR hoặc bỏ tham số thứ 2)
image = cv2.imread("./data/exe_3.webp", cv2.IMREAD_COLOR)
image_float = image.astype(np.float32)

# 2. Thêm Gauss noise (std=50) cho toàn bộ 3 kênh màu (B, G, R)
np.random.seed(42)
noise = np.random.normal(0, 50, image_float.shape)
image_noisy = np.clip(image_float + noise, 0, 255).astype(np.uint8)

# 3. Xử lý phục hồi ảnh màu
# Bước 1: Dùng Median cho từng kênh màu (OpenCV tự làm)
image_median = cv2.medianBlur(image_noisy, 5)

# Bước 2: Dùng NLM chuyên dụng cho ảnh màu (fastNlMeansDenoisingColored)
# h: độ mạnh khử nhiễu kênh Luma, hColor: độ mạnh khử nhiễu kênh Chroma
image_denoised = cv2.fastNlMeansDenoisingColored(image_median, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

# 4. Tính MSE trên ảnh Màu (so sánh trên cả 3 kênh)
mse = np.mean((image_float - image_denoised.astype(np.float32)) ** 2)

print(f"MSE (Anh Mau): {mse:.2f}")

# Hiển thị
cv2.imshow('Anh Goc Color', image)
cv2.imshow('Anh Nhieu Color', image_noisy)
cv2.imshow('Anh Phuc Hoi Color', image_denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Đọc ảnh màu dưới dạng ma trận 3 chiều
# image = cv2.imread("./data/exe_3.webp", cv2.IMREAD_COLOR)

# # Chuyển sang float32
# image_float = image.astype(np.float32)

# # Mô phỏng chụp cùng một ảnh N lần với nhiễu Gaussian
# N = 5000
# # Độ mạnh của Gaussian Noise
# std_noise = 50

# # Mâm thép khổng lồ để cộng dồn N ảnh màu
# # Dùng float64 để tránh tràn số khi cộng N lần (dù giá trị vượt 255)
# sum_images = np.zeros(image_float.shape, dtype=np.float64)

# print(f"Đang xử lý averaging {N} khung hình màu...")

# # Vòng lặp cộng dồn
# for i in range(N):
#     noise = np.random.normal(0, std_noise, image_float.shape)
#     sum_images += (image_float + noise)

# # Lấy trung bình cộng và Clip 1 lần duy nhất ở cuối
# image_averaged = sum_images / N
# image_denoised = np.clip(image_averaged, 0, 255).astype(np.uint8)

# # Tính MSE trên ảnh MÀU
# mse = np.mean((image_float - image_denoised.astype(np.float32)) ** 2)

# print(f"MSE đạt được: {mse:.4f}")

# # Hiển thị
# cv2.imshow('Anh Goc', image)
# cv2.imshow('Anh Phuc Hoi', image_denoised)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
