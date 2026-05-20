import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DỮ LIỆU ĐẦU VÀO VÀ HÀM CHUẨN HÓA Z-SCORE
# ==========================================
# X: Cột 0 là Diện tích (m2), Cột 1 là Số phòng ngủ
X_train = np.array([
    [50, 1],
    [60, 2],
    [80, 2],
    [100, 3],
    [120, 3]
])
# y: Giá nhà (Tỷ VND)
y_train = np.array([1.5, 2.0, 2.6, 3.1, 4.0])

def z_score_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# ==========================================
# 2. CÁC HÀM CỐT LÕI CỦA LINEAR REGRESSION
# ==========================================
def predict(x, w, b):
    return np.dot(x, w) + b

def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, w_in, b_in, learning_rate, num_iters):
    m = X.shape[0]
    w = w_in.copy() # Copy để không đè lên giá trị gốc
    b = b_in
    cost_history = []

    for i in range(num_iters):
        predictions = predict(X, w, b)
        
        # Tính đạo hàm
        dj_dw = (1 / m) * np.dot(X.T, (predictions - y))
        dj_db = (1 / m) * np.sum(predictions - y)
        
        # Cập nhật trọng số
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Bước {i:4d}: Lỗi MSE = {cost:.4f}")
            
    return w, b, cost_history

# ==========================================
# 3. CHẠY THUẬT TOÁN VÀ XEM KẾT QUẢ
# ==========================================
if __name__ == "__main__":
    # Chuẩn hóa dữ liệu
    X_norm, mu, sigma = z_score_normalize(X_train)

    # Khởi tạo thông số
    initial_w = np.zeros(X_norm.shape[1])
    initial_b = 0.0
    iterations = 1000
    alpha = 0.1 # Learning rate

    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    w_final, b_final, J_history = gradient_descent(X_norm, y_train, initial_w, initial_b, alpha, iterations)

    print("\n--- KẾT QUẢ SAU KHI HỌC ---")
    print(f"Trọng số w tìm được: {w_final}")
    print(f"Hệ số b tìm được: {b_final}")

    # Vẽ biểu đồ
    print("\n[HỆ THỐNG] Đang mở biểu đồ Learning Curve...")
    print("=> VUI LÒNG TẮT CỬA SỔ BIỂU ĐỒ ĐỂ CHƯƠNG TRÌNH CHẠY TIẾP PHẦN DỰ ĐOÁN <=")
    
    plt.plot(J_history)
    plt.title("Qua trinh di xuong day bat (Learning Curve)")
    plt.xlabel("So buoc lap (Iterations)")
    plt.ylabel("Do loi MSE (Cost)")
    plt.show() # Code sẽ TẠM DỪNG ở đây cho đến khi bạn tắt cửa sổ popup

    # Dự đoán căn nhà mới
    x_new = np.array([90, 2])
    x_new_norm = (x_new - mu) / sigma 
    gia_du_doan = predict(x_new_norm, w_final, b_final)
    
    print(f"\n=> AI dự đoán giá căn nhà 90m2, 2 phòng ngủ là: {gia_du_doan:.2f} Tỷ VND")
    print("--- HOÀN THÀNH ---")