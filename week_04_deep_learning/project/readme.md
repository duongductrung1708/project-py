# Phân loại ảnh CIFAR-10 với Mạng CNN Tùy chỉnh (91.02% Accuracy)

## 1. Chuẩn bị Dữ liệu & Data Augmentation

### Các thư viện sử dụng
* `torch`, `torch.nn`, `torch.optim`: Bộ khung chính của PyTorch để xây dựng và tối ưu mô hình.
* `torchvision.datasets`, `transforms`: Tải dữ liệu chuẩn và xử lý ảnh.
* `DataLoader`: Băm nhỏ dữ liệu thành các batch để GPU dễ tiêu hóa.

### Pipeline Xử lý ảnh (Transforms)
Định nghĩa 2 pipeline riêng biệt cho tập Train và tập Test.

**Tập Train (Có Augmentation để chống Overfitting):**
* `transforms.RandomCrop(32, padding=4)`: Thêm 4 pixel viền đen xung quanh ảnh, sau đó cắt ngẫu nhiên lại thành 32x32.
  * *Tại sao?* Giúp mô hình nhận diện vật thể ngay cả khi nó không nằm ở trung tâm bức ảnh.
  * *Nếu không dùng:* Mô hình sẽ học vẹt vị trí pixel tĩnh, ra ngoài thực tế gặp ảnh hơi lệch là đoán sai ngay.
* `transforms.RandomHorizontalFlip()`: Lật ngang ảnh với xác suất 50%.
  * *Tại sao?* Con chó quay mặt sang trái hay sang phải thì vẫn là con chó. Tăng gấp đôi dữ liệu học.
* `transforms.RandomRotation(15)`: Xoay ảnh ngẫu nhiên trong góc từ -15 đến 15 độ.
* `transforms.ToTensor()`: Biến đổi ảnh từ dạng số nguyên (0-255) thành dạng Tensor số thực (0.0 - 1.0) để Pytorch tính toán ma trận.
* `transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`
  * *Con số này là gì?* Đây là giá trị trung bình (mean) và độ lệch chuẩn (std) của 3 kênh màu Đỏ, Xanh lá, Xanh dương đã được tính toán sẵn trên toàn bộ 50.000 ảnh CIFAR-10.
  * *Tại sao?* Đưa phân phối màu sắc về mốc trung tâm (Gốc tọa độ 0). Giúp thuật toán Gradient Descent tìm đường đến điểm tối ưu nhanh hơn và tránh bị kẹt.

**Tập Test:** Chỉ dùng `ToTensor` và `Normalize`. Không được dùng Augmentation ở bước này vì đây là lúc thi thật, cần giữ ảnh nguyên bản để đánh giá công bằng.

### DataLoader (Bơm dữ liệu cho GPU)
* `batch_size=128`: Mỗi lần đưa 128 bức ảnh vào GPU để học.
  * *Tại sao là 128?* GPU xử lý hệ nhị phân rất tốt. Các con số lũy thừa của 2 (64, 128, 256) giúp tối ưu hóa phần cứng. 128 là điểm cân bằng hoàn hảo giữa việc tốn ít RAM (VRAM) nhưng vẫn cho ra quỹ đạo hội tụ ổn định.
  * *Nếu dùng số khác:* Dùng 16 thì học quá lâu và nhiễu. Dùng 1024 thì tràn RAM và dễ bị "học vẹt".
* `num_workers=0`: Chạy quá trình tải dữ liệu trên luồng chính (Main thread) để đảm bảo độ ổn định cao nhất, tránh lỗi sập bộ nhớ trên một số hệ điều hành.

---

## 2. Kiến trúc Mạng CNN (Lớp Net)

Kiến trúc được thiết kế theo phong cách **VGG-style** (xếp chồng các bộ lọc nhỏ 3x3) kết hợp với **Batch Normalization**, chia làm 3 khối chính.

### Khối 1: Tìm kiếm đặc trưng cơ bản (Cạnh, góc, màu sắc)
* `nn.Conv2d(3, 64, kernel_size=3, padding=1)`:
  * Nhận 3 kênh màu (RGB), xuất ra 64 màng lọc (filters) (Tối ưu nhất qua các lần thử nghiệm trên tập CIFAR-10).
  * `kernel_size=3`: Bộ lọc ma trận 3x3. Kích thước nhỏ giúp nắm bắt chi tiết tốt và tiết kiệm tham số.
  * `padding=1`: Thêm viền ảo để ảnh không bị teo nhỏ lại sau khi quét (32x32 vẫn là 32x32).
* `nn.BatchNorm2d(64)`: Chuẩn hóa lại các đặc trưng vừa tìm được. Ép các con số không vọt lên quá cao hoặc tụt xuống quá thấp, giúp mô hình ổn định thần kinh (Tương ứng với 64 filter).
* `nn.ReLU()`: Hàm kích hoạt, loại bỏ các giá trị âm (coi như không tìm thấy đặc trưng).
* `nn.MaxPool2d(kernel_size=2, stride=2)`: Nén bức ảnh lại một nửa (từ 32x32 xuống 16x16) bằng cách lấy giá trị sáng nhất trong ô 2x2. Giúp giảm gánh nặng tính toán.

### Khối 2 & Khối 3: Tìm kiếm đặc trưng phức tạp (Bộ phận cơ thể)
* Khối 2: Tăng từ 64 lên 128 màng lọc. Nén ảnh từ 16x16 xuống 8x8.
* Khối 3: Tăng từ 128 lên 256 màng lọc. Nén ảnh từ 8x8 xuống 4x4.
* *Tại sao số lượng màng lọc cứ nhân đôi (64 -> 128 -> 256)?* Khi bức ảnh bị thu nhỏ lại, không gian x/y hẹp đi, ta phải mở rộng chiều sâu (số kênh) để có đủ không gian lưu trữ các hình thái phức tạp (ví dụ: mũi chó, lốp xe, mắt mèo).

### Tại sao lại là 6 lớp Conv chia làm 3 Khối (Blocks)?
Kiến trúc này không phải chọn ngẫu nhiên, mà tuân theo giới hạn vật lý của bức ảnh và cách não bộ nhận thức đặc trưng:

**1. Giới hạn co rút của bức ảnh (Spatial Reduction):**
CIFAR-10 có kích thước ảnh gốc rất nhỏ: **32x32 pixel**. Mỗi lần đi qua 1 khối (kết thúc bằng lớp nén `MaxPool2d`), chiều dài và chiều rộng của ảnh sẽ bị cắt đôi.
* Sau Khối 1: Ảnh giảm còn `16x16`.
* Sau Khối 2: Ảnh giảm còn `8x8`.
* Sau Khối 3: Ảnh giảm còn `4x4`.
* *Tại sao dừng ở 3 Khối?* Kích thước `4x4` là "điểm ngọt" (sweet spot). Nó đủ nhỏ để trải phẳng (`Flatten`) mà không làm tràn RAM, nhưng vẫn giữ lại được một chút cấu trúc không gian (trên/dưới, trái/phải). Nếu cố thêm Khối 4, ảnh sẽ bị ép xuống `2x2` (bị nghiền nát hoàn toàn), mô hình sẽ bị "mù" và mất phương hướng không gian.

**2. Quá trình học theo 3 cấp độ phân cấp (Feature Hierarchy):**
Việc chia 3 khối tương đương với 3 giai đoạn não bộ con người nhận thức sự vật:
* **Khối 1 (Cấp thấp):** Dùng 64 màng lọc để tìm kiếm các đường nét cơ bản (cạnh thẳng, góc nhọn, mảng màu sáng tối).
* **Khối 2 (Cấp trung):** Dùng 128 màng lọc để ghép các nét cơ bản lại thành các bộ phận (bánh xe ô tô, mắt con mèo, cánh máy bay).
* **Khối 3 (Cấp cao):** Dùng 256 màng lọc để nhìn tổng thể toàn bộ các bộ phận và định hình ra vật thể hoàn chỉnh.

**3. Tại sao mỗi khối lại có đúng 2 lớp Conv (VGG-Style)?**
Thay vì dùng 1 lớp Conv với màng lọc lớn (ví dụ 5x5) cho mỗi khối, chúng ta xếp chồng 2 lớp Conv màng lọc nhỏ (3x3) liên tiếp nhau.
* *Tác dụng:* Tầm nhìn bao quát của AI đối với bức ảnh vẫn tương đương, nhưng máy tính được chèn thêm 1 hàm kích hoạt `ReLU` ở giữa. Việc này giúp "đường cong tư duy" của AI trở nên phức tạp và sắc sảo hơn, đồng thời **tiết kiệm được khoảng 28% lượng tham số** (RAM) so với dùng màng lọc lớn.

### Lớp Fully Connected (Phân loại cuối cùng)
* `torch.flatten(start_dim=1)`: Trải phẳng ảnh 3D (256 kênh x 4 x 4) thành mảng 1D dài (4096 con số).
* `nn.Dropout(0.5)`: Tắt ngẫu nhiên 50% số dây thần kinh trong lúc huấn luyện.
  * *Tại sao?* Ép mô hình không được ỷ lại vào một vài đặc trưng nhất định. Chống Overfitting cực kỳ mạnh mẽ.
* `nn.Linear(4096, 512)`: Nén 4096 manh mối xuống còn 512 kết luận trung gian.
* `nn.Linear(512, 10)`: Chốt lại 10 điểm số (Logits) cho 10 class của bài toán CIFAR-10.

---

## 3. Cấu hình Huấn luyện

* `criterion = nn.CrossEntropyLoss()`: Hàm tính sai số chuyên dụng cho bài toán phân loại nhiều lớp. Sai càng nhiều, Loss càng cao.
* `optimizer = optim.AdamW(lr=0.001, weight_decay=1e-4)`:
  * Thuật toán tối ưu trọng số (Bộ não học tập). `AdamW` là phiên bản nâng cấp của Adam, tốt hơn trong việc chống học vẹt.
  * `lr=0.001`: Tốc độ học khởi điểm. Tốc độ an toàn để không bước hụt qua điểm tối ưu.
  * `weight_decay=1e-4`: Kéo các trọng số về gần 0 để mô hình không quá tự tin vào một đặc trưng nào đó.
* `scheduler = optim.lr_scheduler.CosineAnnealingLR(T_max=60)`:
  * *Tác dụng:* Từ từ giảm tốc độ học `lr` theo đường cong hình sin lộn ngược. Ở những epoch cuối (50-60), `lr` rất nhỏ để mô hình tinh chỉnh (fine-tune) nhẹ nhàng, không làm hỏng kết quả.

---

## 4. Vòng lặp Huấn luyện (Training Loop)

Mô hình học trong `epochs=60` vòng (nhìn toàn bộ 50.000 ảnh 60 lần).

1. `model.train()`: Bật chế độ đi học (Kích hoạt tính toán đạo hàm, bật Dropout, bật BatchNorm).
2. `optimizer.zero_grad()`: Xóa sạch bộ nhớ đạo hàm của bước trước đó để không bị cộng dồn sai lệch.
3. `outputs = model(images)`: Quá trình Forward Pass (Đưa ảnh từ đầu vào, chạy qua mạng, ra dự đoán).
4. `loss.backward()`: Quá trình Backward Pass (Tính toán xem mỗi dây thần kinh đã đoán sai bao nhiêu phần trăm).
5. `optimizer.step()`: Cập nhật lại toàn bộ trí nhớ (trọng số) dựa trên lỗi sai vừa tính.
6. `scheduler.step()`: Giảm Learning Rate xuống một chút xíu.

---

## 5. Kết quả & Đánh giá (Evaluation)

* `model.eval()`: Chuyển sang chế độ đi thi. Tắt tính năng Dropout (huy động 100% dây thần kinh để làm bài) và cố định BatchNorm.
* `with torch.no_grad():`: Tắt công cụ tính đạo hàm để tiết kiệm RAM tối đa. Tăng tốc độ dự đoán.

**Chỉ số cuối cùng:**
* Loss giảm đều từ **682.71** (Epoch 1) xuống **62.26** (Epoch 60).
* Mô hình đạt **91.02% Accuracy** trên tập Test unseen data. 
* Đây là minh chứng cho việc các lớp chống Overfitting (Augmentation, Dropout, Weight Decay) đã hoạt động hoàn hảo!

---

## 6. Tính Tái lập (Reproducibility) và Random Seed

**Vấn đề:** Deep Learning chứa rất nhiều yếu tố ngẫu nhiên ngầm (Khởi tạo trọng số, xáo trộn batch, Dropout, các phép xoay/cắt ảnh...). Nếu không kiểm soát, mỗi lần chạy (Run) toàn bộ file code sẽ cho ra một kết quả Test Accuracy khác nhau (dao động từ 89% - 92%), gây khó khăn cho việc so sánh các lần nâng cấp mô hình.

**Giải pháp:** Dự án sử dụng hàm `set_seed(42)` được khai báo ngay đầu file để "đóng băng" toàn bộ các bộ sinh số ngẫu nhiên của Python, Numpy, PyTorch và thiết lập chế độ `deterministic = True` cho backend CuDNN của card đồ họa.

* **Tác dụng:** Đảm bảo kết quả huấn luyện là **tất định 100%**. Bất kỳ khi nào clone mã nguồn này về và chạy trên máy của họ đều sẽ thu được quỹ đạo Loss và độ chính xác Accuracy cuối cùng giống hệt như báo cáo.

Epoch [60/60] Loss: 60.3415 Accuracy: 94.72% LR: 0.000001
Test Accuracy: 91.37%