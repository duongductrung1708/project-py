import numpy as np
import matplotlib.pyplot as plt  # Thêm thư viện trực quan hóa dữ liệu

# 1. Khởi tạo dữ liệu
np.random.seed(42) 
X0 = np.random.randn(50, 2) + [2, 2] # Tạo 50 điểm Class 0 quanh tâm (2,2)
X1 = np.random.randn(50, 2) + [5, 5] # Tạo 50 điểm Class 1 quanh tâm (5,5)

X = np.vstack([X0, X1])
y = np.vstack([np.zeros((50, 1)), np.ones((50, 1))])

# Xáo trộn dữ liệu
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X, y = X[idx], y[idx]

# 2. Hàm kích hoạt Sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

# 3. Hàm dự đoán xác suất
def predict_proba(X, w, b):
    z = np.dot(X, w) + b 
    return sigmoid(z)

# 4. Huấn luyện mô hình (Bổ sung cơ chế lưu lịch sử loss)
def fit_logistic_regression(X, y, learning_rate, epochs):
    m, n = X.shape 
    w = np.zeros((n, 1)) 
    b = 0.0              
    
    # Mảng lưu lịch sử Loss để vẽ đồ thị Learning Curve
    loss_history = []
    
    for i in range(epochs):
        p = predict_proba(X, w, b) 
        
        dj_dw = (1 / m) * np.dot(X.T, (p - y))
        dj_db = (1 / m) * np.sum(p - y)
        
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        # Tính toán Loss tại mỗi epoch và lưu lại
        p_clip = np.clip(p, 1e-15, 1 - 1e-15)
        cost = -(1/m) * np.sum(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
        loss_history.append(cost)
        
        if i % 100 == 0:
            print(f"Epoch {i}: Loss = {cost:.4f}")
            
    return w, b, loss_history

# Tiến hành gọi hàm huấn luyện
w_final, b_final, loss_history = fit_logistic_regression(X, y, learning_rate=0.1, epochs=1000)
print(f"\nTraining xong! w={w_final.flatten()}, b={b_final:.4f}")

# Khởi tạo một khung tranh lớn gồm 2 đồ thị con nằm cạnh nhau (1 hàng, 2 cột)
plt.figure(figsize=(14, 5))

# --- ĐỒ THỊ 1: LOSS HISTORY (LEARNING CURVE) ---
plt.subplot(1, 2, 1)  # Chọn ô số 1
plt.plot(loss_history, color='green', linewidth=2)
plt.title('Biểu đồ Lịch sử Hàm Lỗi (Loss History)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch (Vòng lặp)', fontsize=10)
plt.ylabel('Giá trị BCE Loss', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)  # Tạo lưới mờ để dễ nhìn xu hướng

# --- ĐỒ THỊ 2: DỮ LIỆU & RANH GIỚI QUYẾT ĐỊNH (DECISION BOUNDARY) ---
plt.subplot(1, 2, 2)  # Chọn ô số 2

# Tách dữ liệu thực tế dựa trên nhãn y để tô màu chấm tròn
# y.flatten() == 0 trả về True tại những vị trí thuộc về class 0
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], color='blue', label='Class 0 (Khỏe mạnh)', edgecolors='k')
plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], color='red', label='Class 1 (Nhiễm bệnh)', edgecolors='k')

# Công thức toán đường biên: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
# Chúng ta sẽ vẽ đường biên này chạy từ min của x1 đến max của x1 trên đồ thị
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_values = np.linspace(x1_min, x1_max, 100)

w1, w2 = w_final[0, 0], w_final[1, 0]
# Tính toán tọa độ trục tung (x2) tương ứng với từng điểm trục hoành (x1) để tạo đường biên thẳng
x2_values = (-w1 * x1_values - b_final) / w2

# Vẽ thước kẻ ranh giới lên đồ thị dạng đường đứt nét màu đen bản to
plt.plot(x1_values, x2_values, color='black', linestyle='--', linewidth=2.5, label='Decision Boundary (Ranh giới)')

# Định dạng đồ thị
plt.title('Ranh giới Quyết định của Mô hình', fontsize=12, fontweight='bold')
plt.xlabel('Đặc trưng X1 (Feature 1)', fontsize=10)
plt.ylabel('Đặc trưng X2 (Feature 2)', fontsize=10)
plt.legend(loc='best') # Hiển thị hộp chú thích màu sắc
plt.grid(True, linestyle='--', alpha=0.4)

# Hiển thị toàn bộ bức tranh đồ thị lên màn hình
plt.tight_layout()
plt.show()

# 1. Tạo ngẫu nhiên 1 bệnh nhân mới đến khám (giả lập giá trị ngẫu nhiên từ 1 đến 6)
# Hàm np.random.uniform(min, max, kích_thước) tạo số thực ngẫu nhiên trong khoảng chỉ định
np.random.seed(None)
X_test = np.random.uniform(1.0, 6.0, (1, 2)) 

# 2. Đóng băng trọng số w, b và đưa dữ liệu mới vào AI để tính xác suất nhiễm bệnh
p_test = predict_proba(X_test, w_final, b_final)[0, 0]

# 3. Áp dụng Business Logic (Threshold = 0.2) để chốt đơn nhãn dự đoán
threshold = 0.2
prediction = 1 if p_test >= threshold else 0

# 4. In kết quả ra màn hình Console để báo cáo
print("\n" + "="*40)
print("             AI INFERENCE REPORT")
print("="*40)
print(f"Tọa độ bệnh nhân mới nhận diện: X1 = {X_test[0,0]:.2f}, X2 = {X_test[0,1]:.2f}")
print(f"-> AI tính toán xác suất nhiễm bệnh: {p_test * 100:.2f}%")
if prediction == 1:
    print("-> KẾT LUẬN: BỆNH NHÂN CÓ NGUY CƠ NHIỄM BỆNH (Class 1) 🔴")
else:
    print("-> KẾT LUẬN: BỆNH NHÂN KHỎE MẠNH AN TOÀN (Class 0) 🔵")
print("="*40)

# 5. VẼ LẠI ĐỒ THỊ 2 VÀ ĐÈ ĐIỂM TEST LÊN ĐỂ KIỂM TRA TRỰC QUAN
plt.figure(figsize=(7, 5))

# Vẽ lại các điểm dữ liệu cũ của tập Train để làm nền
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], color='blue', alpha=0.3, label='Train: Class 0', edgecolors='k')
plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], color='red', alpha=0.3, label='Train: Class 1', edgecolors='k')

# Vẽ lại đường biên quyết định cũ
plt.plot(x1_values, x2_values, color='black', linestyle='--', linewidth=2, label='Decision Boundary')

# Đè điểm dữ liệu mới (X_test) lên đồ thị bằng hình NGÔI SAO TO màu vàng để xem vị trí của nó
plt.scatter(X_test[0, 0], X_test[0, 1], color='gold', marker='*', s=300, label='Bệnh nhân mới cần Test', edgecolors='black', zorder=5)

# Định dạng đồ thị
plt.title('Vị trí của Bệnh nhân mới so với Ranh giới AI', fontsize=12, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()