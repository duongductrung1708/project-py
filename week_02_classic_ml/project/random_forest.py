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

# Đọc file CSV vào DataFrame
df = pd.read_csv("./datasets/train.csv")

print("\n===== INFO DATA =====")
# Cho biết số dòng, kiểu dữ liệu, cột nào bị thiếu
print(df.info())

print("\n===== KIỂM TRA NULL =====")
print(df.isnull().sum())

print("\n===== MÔ TẢ SỐ LIỆU =====")
print(df.describe())

print("\n===== KIỂM TRA DUPLICATES =====")
print(df.duplicated().sum())

# Tên cột target cần dự đoán
target_column = "Heart Disease"

print("\n===== PHÂN BỐ TARGET =====")
print(df[target_column].value_counts())

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

# # =========================
# # IMPORT
# # =========================
# import pandas as pd

# from sklearn.model_selection import (
#     train_test_split,
#     cross_val_score
# )

# from sklearn.impute import SimpleImputer

# from sklearn.compose import ColumnTransformer

# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import OneHotEncoder

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     classification_report
# )

# # =========================
# # LOAD DATA
# # =========================
# df = pd.read_csv("./datasets/train.csv")

# target_column = "Heart Disease"

# X = df.drop(columns=[target_column, "id"])

# y = df[target_column].map({
#     "Absence": 0,
#     "Presence": 1
# })

# # =========================
# # KIỂM TRA IMBALANCED DATA
# # =========================
# print("\n===== TARGET DISTRIBUTION =====")

# print(y.value_counts())

# print("\n===== TARGET DISTRIBUTION (%) =====")

# print(
#     y.value_counts(normalize=True) * 100
# )

# # =========================
# # TRAIN / VALIDATION SPLIT
# # =========================
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # =========================
# # FEATURE GROUPS
# # =========================
# numeric_cols = [
#     "Age",
#     "BP",
#     "Cholesterol",
#     "Max HR",
#     "ST depression"
# ]

# categorical_cols = [
#     "Sex",
#     "Chest pain type",
#     "FBS over 120",
#     "EKG results",
#     "Exercise angina",
#     "Slope of ST",
#     "Number of vessels fluro",
#     "Thallium"
# ]

# # =========================
# # PREPROCESSOR
# # =========================
# numeric_transformer = Pipeline([
#     (
#         "imputer",
#         SimpleImputer(strategy="median")
#     )
# ])

# categorical_transformer = Pipeline([
#     (
#         "encoder",
#         OneHotEncoder(
#             handle_unknown="ignore"
#         )
#     )
# ])

# preprocessor = ColumnTransformer([
#     (
#         "num",
#         numeric_transformer,
#         numeric_cols
#     ),
#     (
#         "cat",
#         categorical_transformer,
#         categorical_cols
#     )
# ])

# # =========================
# # MODEL
# # =========================
# model = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=12,
#     min_samples_split=10,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1
# )

# # =========================
# # PIPELINE
# # =========================
# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", model)
# ])

# # =========================
# # TRAIN
# # =========================
# pipeline.fit(X_train, y_train)

# # =========================
# # PREDICT
# # =========================
# y_pred = pipeline.predict(X_val)

# # =========================
# # ACCURACY
# # =========================
# accuracy = accuracy_score(
#     y_val,
#     y_pred
# )

# print("\n===== ACCURACY =====")
# print(f"{accuracy * 100:.2f}%")

# # =========================
# # CONFUSION MATRIX
# # =========================
# cm = confusion_matrix(
#     y_val,
#     y_pred
# )

# print("\n===== CONFUSION MATRIX =====")
# print(cm)

# # TN FP
# # FN TP

# tn, fp, fn, tp = cm.ravel()

# print("\n===== FALSE NEGATIVE =====")
# print(fn)

# print("\n===== FALSE POSITIVE =====")
# print(fp)

# # =========================
# # CLASSIFICATION REPORT
# # =========================
# print("\n===== CLASSIFICATION REPORT =====")

# print(
#     classification_report(
#         y_val,
#         y_pred,
#         target_names=[
#             "Absence",
#             "Presence"
#         ]
#     )
# )

# # =========================
# # CROSS VALIDATION RECALL
# # =========================
# print("\n===== CROSS VALIDATION =====")

# recall_scores = cross_val_score(
#     pipeline,
#     X,
#     y,
#     cv=5,
#     scoring="recall",
#     n_jobs=-1
# )

# print("\nRecall Scores:")

# print(recall_scores)

# print("\nMean Recall:")

# print(
#     f"{recall_scores.mean() * 100:.2f}%"
# )

# # =========================
# # INFERENCE FUNCTION
# # =========================
# def predict_new_patient(patient_dict):

#     patient_df = pd.DataFrame(
#         [patient_dict]
#     )

#     prediction = pipeline.predict(
#         patient_df
#     )[0]

#     probability = pipeline.predict_proba(
#         patient_df
#     )[0]

#     print("\n===== NEW PATIENT =====")

#     if prediction == 1:
#         print("Prediction: Presence")
#     else:
#         print("Prediction: Absence")

#     print(
#         f"Probability Absence : {probability[0] * 100:.2f}%"
#     )

#     print(
#         f"Probability Presence: {probability[1] * 100:.2f}%"
#     )

# # =========================
# # TEST INFERENCE
# # =========================
# new_patient = {
#     "Age": 60,
#     "Sex": 1,
#     "Chest pain type": 4,
#     "BP": 150,
#     "Cholesterol": 280,
#     "FBS over 120": 1,
#     "EKG results": 2,
#     "Max HR": 120,
#     "Exercise angina": 1,
#     "ST depression": 3.0,
#     "Slope of ST": 2,
#     "Number of vessels fluro": 2,
#     "Thallium": 7
# }

# predict_new_patient(
#     new_patient
# )