import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PHẦN 1: CHUẨN BỊ DỮ LIỆU
# ==========================================
# 1A. Tập dữ liệu huấn luyện (Train - Dùng để học bài)
# (X: [Tuổi, Lương], y: [0: Không mua, 1: Mua])
X_train = np.array([
    [20, 20], [25, 40], [30, 60], [35, 80], [45, 50],
    [22, 25], [40, 40], [28, 90], [50, 60], [33, 45]
])
y_train = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0])

# 1B. Tập dữ liệu kiểm thử (Test - Đề thi thật, máy chưa từng thấy)
# Giả lập 4 khách hàng mới toanh bước vào cửa hàng
X_test = np.array([
    [24, 30],  # Trẻ, lương thấp -> Khả năng: Không mua (0)
    [42, 45],  # Trung niên, lương vừa -> Khả năng: Mua (1)
    [31, 85],  # Trẻ, lương rất cao -> Khả năng: Mua (1)
    [19, 15]   # Sinh viên, lương thấp -> Khả năng: Không mua (0)
])
y_test = np.array([0, 1, 1, 0])  # Đáp án gốc của đề thi

# ==========================================
# PHẦN 2: KHỞI TẠO VÀ GẮN "PHANH"
# ==========================================
tree_model = DecisionTreeClassifier(
    criterion='gini',       
    max_depth=2,            # Phanh 1: Chỉ cho mọc tối đa 2 tầng
    min_samples_split=2,    # Phanh 2: Node có >= 2 mẫu thì mới được chẻ
    random_state=42         
)

# ==========================================
# PHẦN 3: HUẤN LUYỆN (FIT)
# ==========================================
tree_model.fit(X_train, y_train)
print("=== HUẤN LUYỆN XONG CÂY QUYẾT ĐỊNH ===")

# ==========================================
# PHẦN 4: ĐÁNH GIÁ (TRAIN vs TEST)
# ==========================================
# 4A. Chấm điểm trên tập Train (Kiểm tra xem có học vẹt tốt không)
y_pred_train = tree_model.predict(X_train)
print(f"Accuracy trên tập Train (Điểm học tủ): {accuracy_score(y_train, y_pred_train) * 100}%")

# 4B. Chấm điểm trên tập Test (Kiểm tra thực lực thật sự)
y_pred_test = tree_model.predict(X_test)
print(f"Accuracy trên tập Test (Điểm thi thật): {accuracy_score(y_test, y_pred_test) * 100}%")

# ==========================================
# PHẦN 5: KHÁM PHÁ BÊN TRONG CÁI CÂY
# ==========================================
print("\n=== KIẾN TRÚC CÂY ===")
print("Độ sâu thực tế của cây:", tree_model.get_depth())
print("Tổng số nút lá (Leaf nodes):", tree_model.get_n_leaves())
