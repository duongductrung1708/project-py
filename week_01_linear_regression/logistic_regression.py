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

# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. KHỞI TẠO DỮ LIỆU (1 FEATURE)
# # ==========================================
# np.random.seed(42) 

# # Tạo dữ liệu ngẫu nhiên 1 chiều (chỉ lấy 1 cột dữ liệu thay vì 2)
# # Nhóm 0: Các điểm tập trung quanh vị trí 2.0 trên trục số
# X0 = np.random.randn(50, 1) + 2.0 
# # Nhóm 1: Các điểm tập trung quanh vị trí 5.0 trên trục số
# X1 = np.random.randn(50, 1) + 5.0 

# # Gộp lại: X có shape (100, 1) - đúng 1 cột đặc trưng
# X = np.vstack([X0, X1])
# y = np.vstack([np.zeros((50, 1)), np.ones((50, 1))])

# # Xáo trộn dữ liệu (Shuffle)
# idx = np.arange(X.shape[0])
# np.random.shuffle(idx)
# X, y = X[idx], y[idx]

# # ==========================================
# # 2. TOÁN HỌC CỐT LÕI (GIỮ NGUYÊN LOGIC)
# # ==========================================
# def sigmoid(z):
#     z = np.clip(z, -500, 500) 
#     return 1 / (1 + np.exp(-z))

# def predict_proba(X, w, b):
#     # np.dot của ma trận (100, 1) với (1, 1) vẫn hoạt động hoàn hảo
#     z = np.dot(X, w) + b 
#     return sigmoid(z)

# # ==========================================
# # 3. HUẤN LUYỆN MÔ HÌNH
# # ==========================================
# def fit_logistic_regression(X, y, learning_rate, epochs):
#     m, n = X.shape # n lúc này sẽ bằng 1
#     w = np.zeros((n, 1)) # w khởi tạo có shape (1, 1)
#     b = 0.0              
    
#     loss_history = []
    
#     for i in range(epochs):
#         p = predict_proba(X, w, b) 
        
#         dj_dw = (1 / m) * np.dot(X.T, (p - y))
#         dj_db = (1 / m) * np.sum(p - y)
        
#         w = w - learning_rate * dj_dw
#         b = b - learning_rate * dj_db
        
#         p_clip = np.clip(p, 1e-15, 1 - 1e-15)
#         cost = -(1/m) * np.sum(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
#         loss_history.append(cost)
        
#         if i % 100 == 0:
#             print(f"Epoch {i}: Loss = {cost:.4f}")
            
#     return w, b, loss_history

# # Gọi hàm huấn luyện với dữ liệu 1 Feature
# w_final, b_final, loss_history = fit_logistic_regression(X, y, learning_rate=0.1, epochs=1000)
# print(f"\nTraining xong! w={w_final.flatten()}, b={b_final:.4f}")

# # ==========================================
# # 4. TRỰC QUAN HÓA ĐỒ THỊ 1 FEATURE
# # ==========================================
# plt.figure(figsize=(14, 5))

# # --- ĐỒ THỊ 1: LOSS HISTORY (Hội tụ vẫn giống hệt bài toán cũ) ---
# plt.subplot(1, 2, 1)
# plt.plot(loss_history, color='green', linewidth=2)
# plt.title('Biểu đồ Lịch sử Hàm Lỗi (Loss History)', fontsize=12, fontweight='bold')
# plt.xlabel('Epoch')
# plt.ylabel('Giá trị BCE Loss')
# plt.grid(True, linestyle='--', alpha=0.6)

# # --- ĐỒ THỊ 2: ĐƯỜNG CONG SIGMOID VÀ ĐIỂM CHỐT RANH GIỚI ---
# plt.subplot(1, 2, 2)

# # Trục x: Giá trị đặc trưng X. Trục y: Nhãn thực tế (0 hoặc 1) để xếp thành 2 hàng chấm
# plt.scatter(X[y.flatten() == 0], y[y.flatten() == 0], color='blue', label='Class 0 (Khỏe)', alpha=0.7, edgecolors='k')
# plt.scatter(X[y.flatten() == 1], y[y.flatten() == 1], color='red', label='Class 1 (Bệnh)', alpha=0.7, edgecolors='k')

# # Vẽ đường cong dự đoán Sigmoid của mô hình chạy mượt mà từ min đến max
# x_curve = np.linspace(X.min() - 1, X.max() + 1, 300).reshape(-1, 1)
# p_curve = predict_proba(x_curve, w_final, b_final)
# plt.plot(x_curve, p_curve, color='magenta', linewidth=2.5, label='Đường cong dự đoán AI')

# # Tính toán vị trí "Cột mốc điểm" Decision Boundary tại Threshold = 0.5 (Nơi w*x + b = 0)
# # Công thức: x = -b / w
# decision_boundary_x = -b_final / w_final[0, 0]

# # Vẽ một đường thẳng đứng màu đen đứt nét đánh dấu cột mốc biên giới trên trục số
# plt.axvline(x=decision_boundary_x, color='black', linestyle='--', linewidth=2, label=f'Ranh giới Quyết định ({decision_boundary_x:.2f})')

# plt.title('Mô hình Phân loại 1 Đặc trưng', fontsize=12, fontweight='bold')
# plt.xlabel('Đặc trưng X (Kích thước khối u)', fontsize=10)
# plt.ylabel('Xác suất dự đoán p / Nhãn thực tế y', fontsize=10)
# plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()

# # ==========================================
# # 5. CHẠY THỬ NGHIỆM THỰC TẾ (INFERENCE 1 FEATURE)
# # ==========================================
# np.random.seed(None)
# # Chỉ sinh ra 1 giá trị ngẫu nhiên duy nhất cho 1 Đặc trưng
# X_test = np.random.uniform(1.0, 6.0, (1, 1)) 

# # Tính xác suất nhiễm bệnh
# p_test = predict_proba(X_test, w_final, b_final)[0, 0]

# # Chốt đơn kết quả bằng Threshold = 0.2
# threshold = 0.2
# prediction = 1 if p_test >= threshold else 0

# print("\n" + "="*40)
# print("             AI INFERENCE REPORT (1-FEATURE)")
# print("="*40)
# print(f"Chỉ số bệnh nhân mới: X = {X_test[0,0]:.2f}")
# print(f"-> AI tính toán xác suất nhiễm bệnh: {p_test * 100:.2f}%")
# if prediction == 1:
#     print("-> KẾT LUẬN: BỆNH NHÂN CÓ NGUY CƠ NHIỄM BỆNH (Class 1) 🔴")
# else:
#     print("-> KẾT LUẬN: BỆNH NHÂN KHỎE MẠNH AN TOÀN (Class 0) 🔵")
# print("="*40)

# # --- VẼ ĐỒ THỊ KIỂM TRA ĐIỂM TEST THỰC TẾ ---
# plt.figure(figsize=(7, 5))
# plt.scatter(X[y.flatten() == 0], y[y.flatten() == 0], color='blue', alpha=0.3, label='Train: Class 0')
# plt.scatter(X[y.flatten() == 1], y[y.flatten() == 1], color='red', alpha=0.3, label='Train: Class 1')
# plt.plot(x_curve, p_curve, color='magenta', alpha=0.5, label='Đường cong AI')
# plt.axvline(x=decision_boundary_x, color='black', linestyle='--', label='Ranh giới')

# # Đè ngôi sao vàng lên tọa độ: Trục x = Giá trị X_test, Trục y = Xác suất p_test mà mô hình trả về
# plt.scatter(X_test[0, 0], p_test, color='gold', marker='*', s=300, label='Bệnh nhân mới', edgecolors='black', zorder=5)

# plt.title('Vị trí Bệnh nhân mới trên Đường cong Xác suất', fontsize=12, fontweight='bold')
# plt.xlabel('Đặc trưng X')
# plt.ylabel('Xác suất p')
# plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # ===================================================
# # BƯỚC 1: TẠO DỮ LIỆU GIẢ LẬP 3 LỚP (MULTICLASS DATA)
# # ===================================================
# np.random.seed(42)

# # Tạo 3 cụm dữ liệu tách biệt nhau trong không gian 2D (2 Features)
# X0 = np.random.randn(40, 2) + [2, 2]  # Lớp 0: Tâm quanh (2, 2)
# X1 = np.random.randn(40, 2) + [5, 5]  # Lớp 1: Tâm quanh (5, 5)
# X2 = np.random.randn(40, 2) + [8, 2]  # Lớp 2: Tâm quanh (8, 2)

# # Gộp dữ liệu: X có kích thước (120, 2)
# X = np.vstack([X0, X1, X2])

# # Gộp nhãn: y_labels dạng số nguyên (0, 1, 2) kích thước (120, 1)
# y_labels = np.vstack([np.zeros((40, 1), dtype=int), 
#                       np.ones((40, 1), dtype=int), 
#                       np.zeros((40, 1), dtype=int) + 2])

# # Trộn bài (Shuffle) dữ liệu
# idx = np.arange(X.shape[0])
# np.random.shuffle(idx)
# X, y_labels = X[idx], y_labels[idx]

# # --- KỸ THUẬT QUAN TRỌNG: ONE-HOT ENCODING MÃ HÓA NHÃN ---
# num_classes = 3
# m = X.shape[0]
# y_onehot = np.zeros((m, num_classes))
# y_onehot[np.arange(m), y_labels.flatten()] = 1  # y_onehot có kích thước (120, 3)

# # ===================================================
# # BƯỚC 2: XÂY DỰNG THUẬT TOÁN SOFTMAX ỔN ĐỊNH SỐ HỌC
# # ===================================================
# def stable_softmax(z):
#     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
#     return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# def predict_proba(X, W, b):
#     z = np.dot(X, W) + b  
#     return stable_softmax(z)

# # ===================================================
# # BƯỚC 3: HUÂN LUYỆN MÔ HÌNH (CATEGORICAL CROSS-ENTROPY)
# # ===================================================
# def fit_softmax_regression(X, y_onehot, learning_rate=0.1, epochs=1000):
#     m, n = X.shape          
#     num_classes = y_onehot.shape[1] 
    
#     W = np.zeros((n, num_classes)) 
#     b = np.zeros((1, num_classes)) 
    
#     loss_history = []
    
#     for i in range(epochs):
#         p = predict_proba(X, W, b) 
        
#         p_clip = np.clip(p, 1e-15, 1 - 1e-15) 
#         loss = -(1 / m) * np.sum(y_onehot * np.log(p_clip))
#         loss_history.append(loss)
        
#         dj_dW = (1 / m) * np.dot(X.T, (p - y_onehot)) 
#         dj_db = (1 / m) * np.sum(p - y_onehot, axis=0, keepdims=True) 
        
#         W = W - learning_rate * dj_dW
#         b = b - learning_rate * dj_db
        
#         if i % 200 == 0:
#             print(f"Epoch {i}: Categorical Cross-Entropy Loss = {loss:.4f}")
            
#     return W, b, loss_history

# # Chạy huấn luyện mô hình đa lớp
# W_final, b_final, loss_history = fit_softmax_regression(X, y_onehot, learning_rate=0.1, epochs=1000)

# # ===================================================
# # BƯỚC 4: KIỂM CHỨNG THỰC TẾ (INFERENCE PHÂN LOẠI)
# # ===================================================
# np.random.seed(None) # Tháo xích số ngẫu nhiên
# X_test = np.random.uniform(1.0, 9.0, (1, 2)) # Sinh ngẫu nhiên 1 tọa độ điểm test mới

# p_test = predict_proba(X_test, W_final, b_final) 
# predicted_class = np.argmax(p_test)

# print("\n" + "="*45)
# print("             AI MULTICLASS REPORT")
# print("="*45)
# print(f"Tọa độ vật thể mới nhận diện: X1 = {X_test[0,0]:.2f}, X2 = {X_test[0,1]:.2f}")
# print(f"-> Xác suất lớp ĐỎ   (Class 0): {p_test[0,0]*100:.2f}%")
# print(f"-> Xác suất lớp XANH (Class 1): {p_test[0,1]*100:.2f}%")
# print(f"-> Xác suất lớp VÀNG (Class 2): {p_test[0,2]*100:.2f}%")
# class_names = ["LỚP ĐỎ (🔴)", "LỚP XANH LÁ (🟢)", "LỚP VÀNG (🟡)"]
# print(f"-> KẾT LUẬN CUỐI CÙNG: Vật thể thuộc về {class_names[predicted_class]}")
# print("="*45)

# # ===================================================
# # BƯỚC 5: TRỰC QUAN HÓA BẰNG PLT (ĐỒ THỊ ĐA LỚP)
# # ===================================================
# plt.figure(figsize=(14, 5))

# # --- ĐỒ THỊ 1: LỊCH SỬ HÀM LỖI (HỘI TỤ ĐA LỚP) ---
# plt.subplot(1, 2, 1)
# plt.plot(loss_history, color='green', linewidth=2)
# plt.title('Lịch sử Hàm Lỗi Đa Lớp (CCE Loss History)', fontsize=12, fontweight='bold')
# plt.xlabel('Epoch')
# plt.ylabel('Giá trị CCE Loss')
# plt.grid(True, linestyle='--', alpha=0.5)

# # --- ĐỒ THỊ 2: BẢN ĐỒ VÙNG QUYẾT ĐỊNH (DECISION REGIONS) ---
# plt.subplot(1, 2, 2)

# # Mẹo quét toàn bộ mặt phẳng Oxy bằng lưới điểm mờ mịn để tô màu vùng lãnh thổ
# x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
#                          np.arange(x2_min, x2_max, 0.02))

# # Dự đoán lớp cho toàn bộ các điểm li ti trên lưới nền Oxy
# grid_points = np.c_[xx1.ravel(), xx2.ravel()]
# grid_proba = predict_proba(grid_points, W_final, b_final)
# grid_predictions = np.argmax(grid_proba, axis=1).reshape(xx1.shape)

# # Đổ màu nền cho 3 vùng lãnh thổ riêng biệt (Đỏ nhạt, Xanh nhạt, Vàng nhạt)
# from matplotlib.colors import ListedColormap
# cmap_background = ListedColormap(['#ffcccc', '#ccffcc', '#fffa0040']) # 40% độ trong suốt cho vàng
# plt.contourf(xx1, xx2, grid_predictions, alpha=0.6, cmap=cmap_background)

# # Vẽ đè các chấm tròn dữ liệu Train thực tế lên để xem mức độ phân tách
# y_flat = y_labels.flatten()
# plt.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], color='red', label='Train: Lớp Đỏ', edgecolors='k')
# plt.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], color='green', label='Train: Lớp Xanh Lá', edgecolors='k')
# plt.scatter(X[y_flat == 2, 0], X[y_flat == 2, 1], color='orange', label='Train: Lớp Vàng', edgecolors='k')

# # Đè NGÔI SAO VÀNG to bản chứa điểm X_test mới lên để xem nó đang đứng ở lãnh thổ quốc gia nào
# plt.scatter(X_test[0, 0], X_test[0, 1], color='gold', marker='*', s=350, label='Vật thể mới cần Test', edgecolors='black', zorder=10)

# # Định dạng thẩm mỹ đồ thị
# plt.title('Bản đồ Vùng Quyết định của Mô hình Softmax', fontsize=12, fontweight='bold')
# plt.xlabel('Feature 1 (Trục hoành X1)')
# plt.ylabel('Feature 2 (Trục tung X2)')
# plt.legend(loc='lower left')
# plt.grid(True, linestyle='--', alpha=0.3)

# plt.tight_layout()
# plt.show()