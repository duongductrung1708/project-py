import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. LOAD DATASET
# =====================================================

digits = load_digits()

X = digits.data
y = digits.target

print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================================
# 2. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 3. GRID SEARCH
# =====================================================

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1],
    "kernel": ["rbf"]
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("\nTraining GridSearchCV...\n")

grid.fit(X_train, y_train)

# =====================================================
# 4. BEST MODEL
# =====================================================

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Score:")
print(f"{grid.best_score_:.4f}")

best_model = grid.best_estimator_

# =====================================================
# 5. TEST PREDICTION
# =====================================================

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(f"{acc:.4f}")

# =====================================================
# 6. CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# 7. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=digits.target_names,
    yticklabels=digits.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

# =====================================================
# 8. TÌM CÁC CẶP BỊ NHẦM NHIỀU NHẤT
# =====================================================

cm_errors = cm.copy()

# Loại bỏ đường chéo chính
np.fill_diagonal(cm_errors, 0)

# Chuyển thành DataFrame để dễ xử lý
error_pairs = []

for actual in range(cm_errors.shape[0]):
    for predicted in range(cm_errors.shape[1]):

        if cm_errors[actual, predicted] > 0:

            error_pairs.append({
                "Actual": actual,
                "Predicted": predicted,
                "Count": cm_errors[actual, predicted]
            })

error_df = pd.DataFrame(error_pairs)

# Sắp xếp giảm dần theo số lần nhầm
error_df = error_df.sort_values(
    by="Count",
    ascending=False
)

print("\nTop 10 Most Confused Class Pairs:")
print(error_df.head(10))

# =====================================================
# 9. BAR CHART CÁC LỖI LỚN NHẤT
# =====================================================

top_errors = error_df.head(10)

plt.figure(figsize=(12, 6))

labels = [
    f"{row['Actual']} → {row['Predicted']}"
    for _, row in top_errors.iterrows()
]

plt.bar(labels, top_errors["Count"])

plt.title("Top 10 Most Frequent Misclassifications")
plt.xlabel("Actual → Predicted")
plt.ylabel("Number of Errors")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()