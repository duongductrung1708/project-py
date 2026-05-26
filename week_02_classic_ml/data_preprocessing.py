# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer

# df = pd.DataFrame({
#     'Tuoi': [25, np.nan, 30, np.nan, 40],
#     'Luong': [20, 40, np.nan, 80, np.nan]
# })

# # 1. Xóa hàng có ô trống
# df_dropped = df.dropna()
# print("--- Sau khi xóa hàng (dropna) ---")
# print(df_dropped) 

# # 2. Dùng Pandas (đơn giản, nhanh)
# # Lưu ý: df.median() tự động bỏ qua NaN, rất thông minh
# df_filled = df.fillna(df.median()) 
# print("\n--- Sau khi fill bằng Median (Pandas) ---")
# print(df_filled)

# # 3. Dùng Scikit-Learn (Chuẩn Production Pipeline)
# imputer = SimpleImputer(strategy='median')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# print("\n--- Sau khi dùng SimpleImputer (Scikit-Learn) ---")
# print(df_imputed)

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Dữ liệu ví dụ: Danh sách khách hàng
data = {
    'GioiTinh': ['Nam', 'Nu', 'Nu', 'Nam', 'Khac'],
    'ThanhPho': ['Ha Noi', 'Da Nang', 'Ha Noi', 'HCM', 'Da Nang']
}
df = pd.DataFrame(data)

print("--- Dữ liệu gốc ---")
print(df)

# ==========================================
# 1. LABEL ENCODING (Cho dữ liệu có thứ tự hoặc chỉ có 2 nhóm)
# Ví dụ: Nam/Nu -> 0/1
# ==========================================
le = LabelEncoder()
df['GioiTinh_Encoded'] = le.fit_transform(df['GioiTinh'])

print("\n--- Sau khi Label Encoding (GioiTinh) ---")
print(df[['GioiTinh', 'GioiTinh_Encoded']])
# Lưu ý: Máy tính có thể hiểu sai là Khac (2) > Nu (1) > Nam (0)


# ==========================================
# 2. ONE-HOT ENCODING (Cho dữ liệu KHÔNG có thứ tự)
# Ví dụ: Ha Noi, HCM, Da Nang không ai lớn hơn ai
# ==========================================
# sparse_output=False để trả về ma trận số thay vì ma trận thưa (dễ đọc hơn)
ohe = OneHotEncoder(sparse_output=False)

# Tạo các cột mới cho Thành phố
city_encoded = ohe.fit_transform(df[['ThanhPho']])
city_names = ohe.get_feature_names_out(['ThanhPho'])

# Đưa vào DataFrame mới
df_encoded = pd.DataFrame(city_encoded, columns=city_names)
df_final = pd.concat([df, df_encoded], axis=1)

print("\n--- Sau khi One-Hot Encoding (ThanhPho) ---")
print(df_final.drop(columns=['ThanhPho']))
