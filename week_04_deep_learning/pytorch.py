import cv2
import torch

# Bước 1: Đọc ảnh bằng OpenCV (Kết quả trả về là NumPy Array chuẩn BGR)
image_bgr = cv2.imread("./tesla_logo.jpg")

# Bước 2: Sửa lỗi của OpenCV, chuyển BGR thành RGB chuẩn
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Bước 3: Đưa NumPy Array vào không gian của PyTorch (Share memory tiết kiệm RAM)
tensor_image = torch.from_numpy(image_rgb)

# Bước 4: Đảo chiều từ HWC (OpenCV) sang CHW (PyTorch)
tensor_chw = tensor_image.permute(2, 0, 1)

# Bước 5: Chuẩn hóa (Normalize) về dải [0, 1]
tensor_final = tensor_chw / 255.0

# Bước 6: Kiểm tra phần cứng và chuyển lên GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_ready = tensor_final.to(device)

# Bước 7: Báo cáo kết quả cuối cùng cho sếp
print("Shape cuối cùng:", tensor_ready.shape)
print("Nơi xử lý hiện tại:", tensor_ready.device)