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

    # In kết quả sau mỗi epoch
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}% "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
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

# Epoch [1/60] Loss: 682.7136 Accuracy: 35.07%LR: 0.001000
# Epoch [2/60] Loss: 537.1317 Accuracy: 50.36%LR: 0.000999
# Epoch [3/60] Loss: 466.4740 Accuracy: 57.70%LR: 0.000997
# Epoch [4/60] Loss: 421.4205 Accuracy: 62.31%LR: 0.000994
# Epoch [5/60] Loss: 383.9298 Accuracy: 65.72%LR: 0.000989
# Epoch [6/60] Loss: 359.6384 Accuracy: 68.22%LR: 0.000983
# Epoch [7/60] Loss: 335.3818 Accuracy: 70.67%LR: 0.000976
# Epoch [8/60] Loss: 317.2416 Accuracy: 72.41%LR: 0.000967
# Epoch [9/60] Loss: 295.7631 Accuracy: 74.44%LR: 0.000957
# Epoch [10/60] Loss: 278.2056 Accuracy: 75.98%LR: 0.000946
# Epoch [11/60] Loss: 265.7827 Accuracy: 77.52%LR: 0.000933
# Epoch [12/60] Loss: 252.1447 Accuracy: 78.52%LR: 0.000919
# Epoch [13/60] Loss: 244.2352 Accuracy: 79.41%LR: 0.000905
# Epoch [14/60] Loss: 230.5095 Accuracy: 80.80%LR: 0.000889
# Epoch [15/60] Loss: 221.8481 Accuracy: 81.32%LR: 0.000872
# Epoch [16/60] Loss: 210.0017 Accuracy: 82.25%LR: 0.000854
# Epoch [17/60] Loss: 201.6660 Accuracy: 83.14%LR: 0.000835
# Epoch [18/60] Loss: 194.8357 Accuracy: 83.73%LR: 0.000815
# Epoch [19/60] Loss: 186.9213 Accuracy: 84.52%LR: 0.000794
# Epoch [20/60] Loss: 180.4672 Accuracy: 84.91%LR: 0.000772
# Epoch [21/60] Loss: 172.6788 Accuracy: 85.60%LR: 0.000750
# Epoch [22/60] Loss: 165.3752 Accuracy: 86.07%LR: 0.000727
# Epoch [23/60] Loss: 163.0889 Accuracy: 86.41%LR: 0.000703
# Epoch [24/60] Loss: 157.9768 Accuracy: 86.80%LR: 0.000679
# Epoch [25/60] Loss: 149.2430 Accuracy: 87.50%LR: 0.000655
# Epoch [26/60] Loss: 145.8665 Accuracy: 87.88%LR: 0.000629
# Epoch [27/60] Loss: 140.8285 Accuracy: 88.07%LR: 0.000604
# Epoch [28/60] Loss: 135.4309 Accuracy: 88.63%LR: 0.000578
# Epoch [29/60] Loss: 130.8284 Accuracy: 88.95%LR: 0.000552
# Epoch [30/60] Loss: 125.1504 Accuracy: 89.51%LR: 0.000526
# Epoch [31/60] Loss: 120.3365 Accuracy: 89.78%LR: 0.000500
# Epoch [32/60] Loss: 118.3269 Accuracy: 89.94%LR: 0.000474
# Epoch [33/60] Loss: 112.5515 Accuracy: 90.47%LR: 0.000448
# Epoch [34/60] Loss: 108.7452 Accuracy: 90.82%LR: 0.000422
# Epoch [35/60] Loss: 106.9869 Accuracy: 90.88%LR: 0.000396
# Epoch [36/60] Loss: 101.7450 Accuracy: 91.16%LR: 0.000371
# Epoch [37/60] Loss: 97.1260 Accuracy: 91.70%LR: 0.000345
# Epoch [38/60] Loss: 96.2259 Accuracy: 91.71%LR: 0.000321
# Epoch [39/60] Loss: 89.0087 Accuracy: 92.32%LR: 0.000297
# Epoch [40/60] Loss: 90.5617 Accuracy: 92.30%LR: 0.000273
# Epoch [41/60] Loss: 87.0091 Accuracy: 92.53%LR: 0.000250
# Epoch [42/60] Loss: 85.2585 Accuracy: 92.70%LR: 0.000228
# Epoch [43/60] Loss: 81.9008 Accuracy: 92.92%LR: 0.000206
# Epoch [44/60] Loss: 79.8141 Accuracy: 93.16%LR: 0.000185
# Epoch [45/60] Loss: 76.8900 Accuracy: 93.47%LR: 0.000165
# Epoch [46/60] Loss: 75.6861 Accuracy: 93.46%LR: 0.000146
# Epoch [47/60] Loss: 72.9269 Accuracy: 93.60%LR: 0.000128
# Epoch [48/60] Loss: 71.1708 Accuracy: 93.85%LR: 0.000111
# Epoch [49/60] Loss: 70.4553 Accuracy: 93.85%LR: 0.000095
# Epoch [50/60] Loss: 68.1443 Accuracy: 94.03%LR: 0.000081
# Epoch [51/60] Loss: 65.8028 Accuracy: 94.25%LR: 0.000067
# Epoch [52/60] Loss: 66.4551 Accuracy: 94.16%LR: 0.000054
# Epoch [53/60] Loss: 64.1856 Accuracy: 94.47%LR: 0.000043
# Epoch [54/60] Loss: 65.0463 Accuracy: 94.32%LR: 0.000033
# Epoch [55/60] Loss: 63.8808 Accuracy: 94.57%LR: 0.000024
# Epoch [56/60] Loss: 63.0544 Accuracy: 94.45%LR: 0.000017
# Epoch [57/60] Loss: 63.4248 Accuracy: 94.42%LR: 0.000011
# Epoch [58/60] Loss: 63.9659 Accuracy: 94.49%LR: 0.000006
# Epoch [59/60] Loss: 62.9502 Accuracy: 94.51%LR: 0.000003
# Epoch [60/60] Loss: 62.2668 Accuracy: 94.60%LR: 0.000001

# Test Accuracy: 91.02%