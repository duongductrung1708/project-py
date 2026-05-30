import pandas as pd # Đọc CSV, xử lý bảng dữ liệu, giống Excel nhưng bằng Python

from sklearn.model_selection import train_test_split # Chia data thành tập học (train) và tập kiểm tra (validation)
from sklearn.impute import SimpleImputer # Điền dữ liệu bị thiếu (NaN)
from sklearn.preprocessing import OneHotEncoder # Biến dữ liệu chữ thành số

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("./datasets/train.csv") # Đọc file CSV vào DataFrame

print("===== 5 DÒNG ĐẦU =====")
print(df.head())

print("\n===== INFO DATA =====")
print(df.info()) # Cho biết số dòng, kiểu dữ liệu, cột nào bị thiếu

print("\n===== DESCRIBE DATA =====")
print(df.describe()) # xem thống kê mean, std, min, max

print("\n===== KIỂM TRA NULL =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

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

print("\n===== SHAPE =====")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# Tìm cột số và cột chữ
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

numeric_cols = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression"
]

print("\n===== NUMERIC COLUMNS =====")
print(numeric_cols)

print("\n===== CATEGORICAL COLUMNS =====")
print(categorical_cols)

# Xử lí null cho cột số
num_imputer = SimpleImputer(strategy='median')

X_train[numeric_cols] = num_imputer.fit_transform(
    X_train[numeric_cols]
)

# Chỉ transform cho validation
X_val[numeric_cols] = num_imputer.transform(
    X_val[numeric_cols]
)

# One hot encoding cho cột chữ
encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

# Fit trên train
encoded_train = encoder.fit_transform(
    X_train[categorical_cols]
)

# Chỉ transform trên val
encoded_val = encoder.transform(
    X_val[categorical_cols]
)

# Chuyển thành DataFrame
encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)

encoded_val_df = pd.DataFrame(
    encoded_val,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_val.index
)

# Xóa cột categorical cũ
X_train = X_train.drop(columns=categorical_cols)
X_val = X_val.drop(columns=categorical_cols)

# Ghép lại
X_train = pd.concat([X_train, encoded_train_df], axis=1)
X_val = pd.concat([X_val, encoded_val_df], axis=1)

# Train base model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_val)

# Tính accuracy
accuracy = accuracy_score(y_val, y_pred)

print("\n===== ACCURACY =====")
print(f"Accuracy: {accuracy * 100:.2f}%")