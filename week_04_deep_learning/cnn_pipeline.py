import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Đọc ảnh bằng OpenCV (Kết quả trả về là NumPy Array chuẩn BGR)
image_bgr = cv2.imread("./tesla_logo.jpg")
# Sửa lỗi của OpenCV, chuyển BGR thành RGB chuẩn
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Đưa NumPy Array vào không gian của PyTorch (Share memory tiết kiệm RAM)
tensor_image = torch.from_numpy(image_rgb)
# Đảo chiều từ HWC (OpenCV) sang CHW (PyTorch)
tensor_chw = tensor_image.permute(2, 0, 1)
# Chuẩn hóa (Normalize) về dải [0, 1]
tensor_final = tensor_chw / 255.0

# Kiểm tra phần cứng và chuyển lên GPU nếu có
# Cách viết chuẩn cho mọi máy tính (NVIDIA, Mac, Windows)
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon GPU (Mac M1/M2/M3)
else:
    device = "cpu"   # CPU
tensor_ready = tensor_final.to(device)

print("Shape cuối cùng:", tensor_ready.shape)
print("Nơi xử lý hiện tại:", tensor_ready.device)

input_batch = tensor_final.unsqueeze(0)

conv_layer = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
output = conv_layer(input_batch)
print("Output shape:", output.shape)

# Visualize the output
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(output[0, i].cpu().detach().numpy(), cmap='gray')
    axes[row, col].set_title(f'Channel {i}')
plt.show()

relu = nn.ReLU()
output_relu = relu(output)

# Định nghĩa Pooling layers
# kernel_size=2: cửa sổ quét 2x2
# stride=2: bước nhảy là 2 (không bị chồng lấn)
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Chạy pooling
output_max = maxpool(output_relu)

print("MaxPool output shape:", output_max.shape)

# Flatten
output_flatten = output_max.flatten(start_dim=1)

print("Flatten output shape:", output_flatten.shape)

# ==========================================
# Classifier
# ==========================================
classifier = nn.Linear(
    in_features=output_flatten.shape[1],
    out_features=3
)

logits = classifier(output_flatten)

print("Logits shape:", logits.shape)

# ==========================================
# Softmax
# ==========================================
softmax = nn.Softmax(dim=1)
probabilities = softmax(logits)

print("Probabilities:", probabilities)

prediction = torch.argmax(
    probabilities,
    dim=1
)

print("Predicted class:", prediction.item())

# ==========================================
# GIẢ LẬP NHÃN THẬT
# ==========================================
target = torch.tensor([1])

print("Ground Truth:", target.item())

# ==========================================
# LOSS FUNCTION
# ==========================================
criterion = nn.CrossEntropyLoss()

loss = criterion(logits, target)

print("Loss:", loss.item())

# ==========================================
# BACKPROPAGATION
# ==========================================
loss.backward()

print("\nGradient của classifier.weight:")
print(classifier.weight.grad)

print("\nGradient của classifier.bias:")
print(classifier.bias.grad)

# ==========================================
# OPTIMIZER
# ==========================================
optimizer = torch.optim.SGD(
    classifier.parameters(),
    lr=0.01
)

print("\nWeight trước update:")
print(classifier.weight.data[0][:5])

optimizer.step()

print("\nWeight sau update:")
print(classifier.weight.data[0][:5])