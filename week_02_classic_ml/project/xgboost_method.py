# Đọc CSV, xử lý bảng dữ liệu
import pandas as pd

# Chia train/validation
from sklearn.model_selection import train_test_split

# Đánh giá
from sklearn.metrics import (
    accuracy_score,
)

# XGBoost
from xgboost import XGBClassifier

df = pd.read_csv("./datasets/train.csv")

target_column = "Heart Disease"

X = df.drop(columns=[target_column, "id"])

y = df[target_column].map({
    "Absence": 0,
    "Presence": 1
})

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 1. KHAI BÁO CÁC CỘT LÀ CATEGORICAL
categorical_cols = [
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# Ép kiểu dữ liệu của Pandas từ 'int64' sang 'category'
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')

# 2. CẬP NHẬT MODEL
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    enable_categorical=True,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\n===== ACCURACY =====")
print(f"Accuracy: {accuracy * 100:.2f}%")