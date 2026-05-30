import cv2
import numpy as np
import urllib.request
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================
# PHẦN 1: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST TRÊN DỮ LIỆU MNIST
# ==============================================================
print("Đang tải dữ liệu MNIST (Bản gốc 28x28) từ server dự phòng của Google...")
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
path = "./resource/mnist.npz"

# Đảm bảo thư mục resource tồn tại
os.makedirs('./resource', exist_ok=True)

# Tải file nếu chưa có trong máy
if not os.path.exists(path):
    print("Đang tải file mnist.npz (khoảng 11MB)...")
    urllib.request.urlretrieve(url, path)
    print("Tải thành công!")

# Đọc file numpy nén
with np.load(path, allow_pickle=True) as f:
    X_train_raw, y_train_raw = f['x_train'], f['y_train']
    X_test_raw, y_test_raw = f['x_test'], f['y_test']

# Gộp data lại để chia theo chuẩn của chúng ta
X_full = np.concatenate((X_train_raw, X_test_raw))
y_full = np.concatenate((y_train_raw, y_test_raw))

# ML BRIDGE QUAN TRỌNG: 
# Dữ liệu hiện tại đang là (70000, 28, 28). Ta phải bẻ dẹp nó thành (70000, 784) cho Random Forest
X = X_full.reshape(X_full.shape[0], 28 * 28)
y = y_full.astype(str) # Chuyển nhãn (Target) sang dạng chuỗi (String) cho an toàn

print(f"Kích thước X sau khi làm phẳng: {X.shape}") # Sẽ in ra (70000, 784)

# Chia tập dữ liệu (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Đang trồng Khu rừng ngẫu nhiên (Training)... sẽ mất vài giây...")
# Triệu hồi Rừng ngẫu nhiên với 100 cái Cây Quyết Định
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ==============================================================
# PHẦN 2: OPENCV PIPELINE (GIỮ NGUYÊN CODE CỦA BẠN - RESIZE VỀ 28x28)
# ==============================================================
def predict_handwritten_digit(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi: Không tìm thấy ảnh!")
        return
        
    _, img_inverted = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Ép kích thước về chuẩn 28x28 (KHÔNG ép thang màu 0-16 nữa)
    img_resized = cv2.resize(img_inverted, (28, 28), interpolation=cv2.INTER_AREA)
    
    feature_vector = img_resized.flatten()
    input_data = np.array([feature_vector])
    prediction = model.predict(input_data)
    
    print(f"🤖 AI Dự đoán bức ảnh này là số: {prediction[0]}")

# Gọi hàm thử nghiệm với bức ảnh số 8 bạn vừa tạo lúc nãy
predict_handwritten_digit("./resource/test_digit.jpg", rf_model)