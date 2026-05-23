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
    mu = np.mean(X, axis=0) # Giá trị trung bình của toàn bộ diện tích và số phòng ngủ các nhà trong tập data.
    print(f"Mean: {mu}")
    sigma = np.std(X, axis=0) # Độ lệch chuẩn của toàn bộ diện tích và số phòng ngủ các nhà trong tập data.
    print(f"Std: {sigma}")
    X_norm = (X - mu) / sigma # Chuẩn hóa dữ liệu
    print(f"Normalized X: {X_norm}")
    return X_norm, mu, sigma

# ==========================================
# 2. CÁC HÀM CỐT LÕI CỦA LINEAR REGRESSION
# ==========================================
def predict(x, w, b):
    # Lấy riêng cột diện tích (cột 0)
    dien_tich = x[:, 0:1] # viết 0:1 để NumPy giữ nguyên dạng ma trận cột dọc (N hàng, 1 cột)
    
    # Tự sinh dữ liệu bậc cao
    dien_tich_mu_2 = dien_tich ** 2
    
    # Lấy cột số phòng ngủ (cột 1)
    phong_ngu = x[:, 1:2]
    phong_ngu_mu_2 = phong_ngu ** 2
    
    # Ráp chúng lại thành một ma trận mở rộng tự động: [Diện tích, Phòng ngủ, Diện tích^2]
    x_poly = np.hstack((dien_tich, phong_ngu, dien_tich_mu_2, phong_ngu_mu_2))
    
    # Nhân ma trận phẳng như bình thường (w lúc này bắt buộc phải có 4 phần tử)
    return np.dot(x_poly, w) + b

def compute_cost(m, y_pred, y_actual):
    cost = (1 / (2 * m)) * np.sum((y_pred - y_actual) ** 2) # Tính lỗi MSE
    return cost

def gradient_descent(X, y, w_in, b_in, learning_rate, num_iters):
    m = X.shape[0] # Số lượng dữ liệu (5)
    w = w_in.copy() # Sao chép w_in thành w
    b = b_in # Sao chép b_in thành b
    cost_history = [] # Lưu lại lịch sử lỗi
    
    epsilon = 1e-6
    cost_old = None 

    # --- ĐOẠN SỬA ĐỒNG BỘ ĐẠO HÀM ---
    # Tái tạo lại ma trận đa thức giống hệt như trong hàm predict để phục vụ tính đạo hàm
    dien_tich = X[:, 0:1]
    phong_ngu = X[:, 1:2]
    
    X_poly = np.hstack((dien_tich, phong_ngu, dien_tich ** 2, phong_ngu ** 2))
    
    for i in range(num_iters):
        # BƯỚC 1: TÍNH DỰ ĐOÁN
        predictions = predict(X, w, b) # Tính giá trị dự đoán
        
        # --- BƯỚC 2: TÍNH COST ---
        # Tính xem tại vị trí w, b hiện tại thì mô hình đang lệch bao nhiêu
        cost_current = compute_cost(m, predictions, y)
        cost_history.append(cost_current)
        
        # --- BƯỚC 2: KIỂM TRA HỘI TỤ (EARLY STOPPING) ---
        # Vòng lặp đầu tiên (i=0) chưa có cost_old nên không so sánh.
        # Từ vòng thứ 2 (i>0), ta lấy cost_old (của vòng trước) trừ đi cost_current (vòng này)
        # if cost_old is not None and abs(cost_old - cost_current) < epsilon:
        #     print(f"Mô hình đã CHẠM ĐÁY (hội tụ) tại bước {i}.")
        #     break
            
        # Cất giá trị cost của vòng này vào biến cost_old để vòng sau dùng so sánh
        cost_old = cost_current 
        
        if i % 100 == 0:
            # Sửa lại dòng print của em thành:
            print(f"Bước {i:4d}: Lỗi MSE = {cost_current:.4f} | Weights={w} | Bias={b:.3f}")
            
        # --- BƯỚC 3: TÍNH ĐẠO HÀM VÀ BƯỚC ĐI (CẬP NHẬT W, B) ---
        dj_dw = (1 / m) * np.dot(X_poly.T, (predictions - y)) # Tính đạo hàm w
        dj_db = (1 / m) * np.sum(predictions - y) # Tính đạo hàm b
        
        # Cập nhật trọng số (Bộ w, b này sẽ được dùng để tính Cost ở ĐẦU vòng lặp tiếp theo)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
            
    return w, b, cost_history

# ==========================================
# 3. CHẠY THUẬT TOÁN VÀ XEM KẾT QUẢ
# ==========================================
if __name__ == "__main__":
    # Chuẩn hóa dữ liệu
    X_norm, mu, sigma = z_score_normalize(X_train)

    # Khởi tạo thông số
    print(f"X_norm.shape[1] = {X_norm.shape[1]}")
    initial_w = np.zeros(4) # Trọng số ban đầu
    initial_b = 0.0 # Hệ số chặn
    iterations = 4600 # Số lần lặp
    alpha = 0.01 # Learning rate
    print(f"initial_w = {initial_w} - initial_b = {initial_b} - iterations = {iterations} - alpha = {alpha}")

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
    x_new = np.array([[90, 2]]) # Viết dạng ma trận 2D để hàm predict cắt slice [:, 0] không bị lỗi
    x_new_norm = (x_new - mu) / sigma 
    gia_du_doan = predict(x_new_norm, w_final, b_final)
    
    print(f"\n=> AI dự đoán giá căn nhà 90m2, 2 phòng ngủ là: {gia_du_doan[0]:.2f} Tỷ VND")
    print("--- HOÀN THÀNH ---")