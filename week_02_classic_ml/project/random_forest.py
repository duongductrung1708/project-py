 # Đọc CSV, xử lý bảng dữ liệu, giống Excel nhưng bằng Python
import pandas as pd
# Chia data thành tập học (train) và tập kiểm tra (validation)
from sklearn.model_selection import train_test_split
# Điền dữ liệu bị thiếu (NaN)
from sklearn.impute import SimpleImputer
# Biến dữ liệu chữ thành số
from sklearn.preprocessing import OneHotEncoder
# Mô hình Random Forest
from sklearn.ensemble import RandomForestClassifier
# Đo độ chính xác
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
# Đọc file CSV vào DataFrame
df = pd.read_csv("./datasets/train.csv")

print("\n===== INFO DATA =====")
# Cho biết số dòng, kiểu dữ liệu, cột nào bị thiếu
print(df.info())

print("\n===== KIỂM TRA NULL =====")
print(df.isnull().sum())

# Tên cột target cần dự đoán
target_column = "Heart Disease"

# Xóa cột target và id
X = df.drop(columns=[target_column, "id"])
# Chuyển đổi target từ string sang số (0/1)
y = df[target_column].map({
    "Absence": 0,
    "Presence": 1
})

# Chia dữ liệu thành tập train và validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_cols = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression"
]

categorical_cols = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium"
]

# Xử lí null cho cột số
num_imputer = SimpleImputer(strategy='median')

X_train[numeric_cols] = num_imputer.fit_transform(
    X_train[numeric_cols]
)

# Chỉ transform cho validation
X_val[numeric_cols] = num_imputer.transform(
    X_val[numeric_cols]
)

# Train base model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Fit model
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_val)

# Tính accuracy
accuracy = accuracy_score(y_val, y_pred)

print("\n===== ACCURACY =====")
print(f"Accuracy: {accuracy * 100:.2f}%")