import cv2
import numpy as np

image = cv2.imread("./data/bai3/exe_3.webp", cv2.IMREAD_COLOR)
# Chuyển ảnh sang float32 để xử lý
image_float = image.astype(np.float32)

# Thêm nhiễu Gaussian
np.random.seed(42)
noise = np.random.normal(0, 50, image_float.shape)
image_noisy = np.clip(image_float + noise, 0, 255).astype(np.uint8)

# Denoise
image_median = cv2.medianBlur(image_noisy, 5)
image_denoised = cv2.fastNlMeansDenoisingColored(image_median, None, h=30, hColor=30, templateWindowSize=7, searchWindowSize=21)

# Chuyển ảnh thành danh sách pixel
pixels = image_denoised.reshape((-1, 3)).astype(np.float32)

# Palette màu chuẩn của logo (BGR)
palette = np.array([
    [2,2,3],
    [65,63,66],
    [126,126,126],
    [211,211,211],
    [244,244,245],
    [102,211,139],
    [68,41,245],
    [244,137,20]
], dtype=np.float32)

# Tính khoảng cách từ từng pixel tới từng màu
distances = np.linalg.norm(
    pixels[:, np.newaxis] - palette[np.newaxis, :],
    axis=2
)

# Lấy màu gần nhất
nearest_idx = np.argmin(distances, axis=1)

# Thay pixel bằng màu gần nhất
quantized = palette[nearest_idx]

# Khôi phục shape ảnh
image_quantized = quantized.reshape(image_denoised.shape).astype(np.uint8)

print("Palette:")
for color in palette.astype(np.uint8):
    print(color)
# Lưu ảnh sau khi quantized
cv2.imwrite("./data/bai3/quantized.webp", image_quantized)
mse = np.mean((image_float - image_quantized.astype(np.float32)) ** 2)

print(f"MSE: {mse:.2f}")

cv2.imshow("Original", image)
cv2.imshow("Noisy", image_noisy)
cv2.imshow("Denoised", image_denoised)
cv2.imshow("KMeans Quantized", image_quantized)

cv2.waitKey(0)
cv2.destroyAllWindows()