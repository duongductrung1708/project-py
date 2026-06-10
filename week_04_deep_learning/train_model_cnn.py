import torch
import torch.nn as nn
import torch.optim as optim

# Dataset có sẵn trong torchvision
from torchvision import datasets
from torchvision import transforms

# Chia dữ liệu thành mini-batch
from torch.utils.data import DataLoader

# Chuyển ảnh PIL Image -> Tensor
# Đồng thời scale từ [0,255] -> [0,1]
transform = transforms.ToTensor()

# Fashion-MNIST train
# 60.000 ảnh
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Fashion-MNIST test
# 10.000 ảnh
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Mỗi lần lấy 64 ảnh
# shuffle=True để tránh học thuộc thứ tự dữ liệu
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Test không cần shuffle
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Định nghĩa kiến trúc CNN (các layer + luồng dữ liệu từ ảnh đầu vào đến kết quả dự đoán)
class Net(nn.Module):

    # Khai báo các layer
    def __init__(self):

        # Khởi tạo bộ khung của PyTorch để model có thể quản lý weight, gradient, optimizer, save/load model và training được
        super().__init__()

        # Lớp Convolution đầu tiên: học 16 bộ lọc 3x3 để trích xuất đặc trưng từ ảnh đầu vào 1 kênh
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Số channel đầu vào
            out_channels=32,    # Số feature maps/kernel sẽ học (32)
            kernel_size=3,      # Kích thước kernel (3x3)
            padding=1           # Thêm viền 0 để giữ nguyên H,W
        )

        # Giúp CNN học được các đặc trưng phi tuyến phức tạp
        self.relu1 = nn.ReLU()  # Biến mọi giá trị âm thành 0, giữ nguyên giá trị dương

        # Giảm kích thước feature map xuống 2 lần, giữ lại đặc trưng quan trọng nhất
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,      # Cửa sổ pooling 2x2
            stride=2            # Bước nhảy 2 pixel
        )

        # Lớp Convolution thứ hai: học 32 bộ lọc 3x3 để trích xuất đặc trưng từ feature maps của conv1
        self.conv2 = nn.Conv2d(
            in_channels=32,     # Số channel đầu vào
            out_channels=64,    # Số feature maps/kernel sẽ học (64)
            kernel_size=3,      # Kích thước kernel (3x3)
            padding=1           # Thêm viền 0 để giữ nguyên H,W
        )

        # Giúp CNN học được các đặc trưng phi tuyến phức tạp
        self.relu2 = nn.ReLU()

        # Tiếp tục giảm kích thước feature map để giảm số lượng tham số
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,      # Cửa sổ pooling 2x2
            stride=2            # Bước nhảy 2 pixel
        )

        # Bộ phân loại cuối cùng: chuyển đặc trưng đã học thành điểm số cho 10 lớp FashionMNIST
        self.fc = nn.Linear(
            64 * 7 * 7,         # Số feature maps sau flatten (64 * 7 * 7 = 3136)
            10                  # Số class (10 lớp)
        )

    def forward(self, x):
        # (64,1,28,28)
        x = self.conv1(x)

        # (64,16,28,28)
        x = self.relu1(x)

        # (64,16,14,14)
        x = self.pool1(x)

        # (64,32,14,14)
        x = self.conv2(x)

        # (64,32,14,14)
        x = self.relu2(x)

        # (64,32,7,7)
        x = self.pool2(x)

        # Chuyển feature maps (64,64,7,7) thành vector (64,3136)
        x = torch.flatten(
            x,
            start_dim=1  # Bắt đầu flatten từ dimension 1 (batch_size, channels, height, width) -> (batch_size, features)
        )

        # Classifier: 3136 đặc trưng -> 10 logits (10 lớp FashionMNIST)
        x = self.fc(x)

        # Trả về logits cho CrossEntropyLoss và dự đoán class
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

# Adam tự động điều chỉnh learning rate
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Số lần quét toàn bộ tập train
epochs = 10

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
        f"Accuracy: {accuracy:.2f}%"
    )

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

# Epoch [1/10] Loss: 432.5802 Accuracy: 83.50%
# Epoch [2/10] Loss: 287.8933 Accuracy: 88.95%
# Epoch [3/10] Loss: 250.6999 Accuracy: 90.44%
# Epoch [4/10] Loss: 226.7229 Accuracy: 91.30%
# Epoch [5/10] Loss: 207.7484 Accuracy: 92.07%
# Epoch [6/10] Loss: 192.4482 Accuracy: 92.52%
# Epoch [7/10] Loss: 179.2946 Accuracy: 93.12%
# Epoch [8/10] Loss: 167.9693 Accuracy: 93.55%
# Epoch [9/10] Loss: 157.1818 Accuracy: 94.01%
# Epoch [10/10] Loss: 146.1042 Accuracy: 94.35%

# Test Accuracy: 91.76%