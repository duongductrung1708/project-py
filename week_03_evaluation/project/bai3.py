import cv2
import numpy as np

# Đọc ảnh và ép kiểu
image = cv2.imread("./data/exe_3.webp", cv2.IMREAD_GRAYSCALE)
image_float = image.astype(np.float32)

# Thêm Gauss noise
np.random.seed(42)
noise = np.random.normal(0, 50, image_float.shape)
image_noisy = np.clip(image_float + noise, 0, 255).astype(np.uint8)

# Xử lý phục hồi
# Dùng Median diệt các điểm nhiễu lốm đốm cực đoan
image_median = cv2.medianBlur(image_noisy, 5)
# Dùng NLM vuốt mịn lại bề mặt và giữ cạnh
image_denoised = cv2.fastNlMeansDenoising(image_median, None, h=10, templateWindowSize=7, searchWindowSize=21)

# Tính MSE
mse = np.mean((image_float - image_denoised.astype(np.float32)) ** 2)

print(f"MSE: {mse:.2f}")

# Hiển thị
cv2.imshow('Anh Goc', image)
cv2.imshow('Anh Nhieu', image_noisy)
cv2.imshow('Anh Phuc Hoi', image_denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()
