import torch
import torch.nn as nn
import torch.optim as optim

# Dataset có sẵn trong torchvision
from torchvision import datasets
from torchvision import transforms

# Chia dữ liệu thành mini-batch
from torch.utils.data import DataLoader

import random
import numpy as np

import time

def set_seed(seed=42):
    # Cố định random của Python
    random.seed(seed)
    
    # Cố định random của Numpy (nếu có dùng)
    np.random.seed(seed)
    
    # Cố định random của PyTorch trên CPU
    torch.manual_seed(seed)
    
    # Cố định random của PyTorch trên GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Tắt các thuật toán tối ưu phi tất định của CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gọi hàm với một con số bất kỳ (thường giới AI hay dùng số 42)
set_seed(42)

# Ghép nhiều phép biến đổi ảnh thành một pipeline
transform = transforms.Compose([
    # Chuyển ảnh PIL -> Tensor và scale pixel từ [0,255] -> [0,1]
    transforms.ToTensor(),
    # Chuẩn hóa dữ liệu ảnh theo mean và std của CIFAR10
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # Giá trị trung bình của kênh R,G,B
        std=(0.2470, 0.2435, 0.2616)    # Độ lệch chuẩn của kênh R,G,B
    )
])  

# Data Augmentation cho tập train để giảm overfitting
train_transform = transforms.Compose([
    # Thêm viền 4 pixel rồi crop ngẫu nhiên ảnh 32x32
    transforms.RandomCrop(
        32,
        padding=4
    ),
    # Lật ngang ảnh với xác suất 50%
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # Chuyển ảnh PIL -> Tensor và scale pixel về [0,1]
    transforms.ToTensor(),
    # Chuẩn hóa dữ liệu theo mean và std của CIFAR10
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Transform cho tập test (không augmentation)
test_transform = transforms.Compose([
    # Chuyển ảnh PIL -> Tensor và scale pixel về [0,1]
    transforms.ToTensor(),
    # Chuẩn hóa dữ liệu theo mean và std của CIFAR10
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# CIFAR10 train
# 50.000 ảnh
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# CIFAR10 test
# 10.000 ảnh
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# Mỗi lần lấy 64 ảnh
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

# Test không cần shuffle
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

# Định nghĩa kiến trúc CNN (các layer + luồng dữ liệu từ ảnh đầu vào đến kết quả dự đoán)
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn4 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv5 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization để ổn định và tăng tốc training
        self.bn6 = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Dropout 50% số neurons để giảm overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layer 1 để giảm chiều dữ liệu
        self.fc1 = nn.Linear(
            256 * 4 * 4, # 256 * 4 * 4 = 4096 = 2^12
            512 # 4096 -> 512 = 2^9
        )

        # Fully Connected Layer 2 để phân loại 10 classes
        self.fc2 = nn.Linear(
            512, # 512 -> 10
            10 # 10 classes
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.pool3(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "mps"
    if torch.backends.mps.is_available()
    else
    "cpu"
)

print("Device:", device)

# Khởi tạo mô hình CNN và chuyển toàn bộ trọng số lên CPU/GPU để huấn luyện
model = Net().to(device)

print(model)

# Bài toán Classification -> CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Adam dùng gradient từ Backpropagation để cập nhật toàn bộ trọng số của CNN
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4 # Kéo các weight về gần 0 một chút để tránh overfitting
)

# Số lần quét toàn bộ tập train
epochs = 60

# Tự động giảm Learning Rate theo đường cong cosine
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

for epoch in range(epochs):

    start_time = time.time()

    # Chuyển model sang chế độ huấn luyện
    model.train()

    # Tổng loss của cả epoch
    running_loss = 0

    # Đếm số dự đoán đúng
    correct = 0

    # Tổng số ảnh đã xử lý
    total = 0

    # Lặp qua từng batch trong train dataset
    for images, labels in train_loader:

        # Chuyển dữ liệu lên CPU/GPU
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradient của epoch trước
        optimizer.zero_grad()

        # Forward Pass: Image → Conv → ReLU → Pool → FC
        outputs = model(images)

        # outputs shape: (batch_size, num_classes)
        loss = criterion(
            outputs,
            labels
        )

        # Backward Pass: Tính gradient bằng Backpropagation
        loss.backward()

        # Cập nhật trọng số theo gradient
        optimizer.step()

        # Cộng dồn loss để theo dõi quá trình học
        running_loss += loss.item()

        # Lấy class có score cao nhất
        _, predicted = torch.max(
            outputs,
            dim=1
        )

        # Cộng số lượng ảnh trong batch
        total += labels.size(0)

        # Đếm số ảnh dự đoán đúng
        correct += (
            predicted == labels
        ).sum().item()

    # Accuracy của toàn bộ epoch
    accuracy = (
        correct / total
    ) * 100

    end_time = time.time()
    epoch_duration = end_time - start_time

    # In kết quả sau mỗi epoch
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}% "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} "
        f"Time: {epoch_duration:.2f}s"
    )

    # Giảm learning rate theo lịch đã định
    scheduler.step()

# Chuyển model sang chế độ đánh giá (Evaluation Mode)
model.eval()

# Đếm số dự đoán đúng
correct = 0

# Tổng số ảnh trong tập test
total = 0

# Tắt tính toán gradient để tiết kiệm RAM và tăng tốc độ
with torch.no_grad():

    # Duyệt qua từng batch trong tập test
    for images, labels in test_loader:

        # Chuyển dữ liệu lên CPU/GPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass: Chỉ dự đoán, không cập nhật trọng số
        outputs = model(images)

        # Lấy class có score cao nhất
        _, predicted = torch.max(
            outputs,
            dim=1
        )

        # Cộng số lượng ảnh đã kiểm tra
        total += labels.size(0)

        # Đếm số ảnh dự đoán đúng
        correct += (
            predicted == labels
        ).sum().item()

# Tính Accuracy trên toàn bộ tập test
test_accuracy = (
    correct / total
) * 100

# In kết quả cuối cùng
print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)
