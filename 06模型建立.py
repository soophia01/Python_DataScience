import pandas as pd
import numpy as np
import time


# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

train_label2 = pd.read_csv("data/train_label2.csv",
                          encoding = "big5")
test_label = pd.read_csv("data/sample_submission.csv",
                          encoding = "big5")
data_label = pd.concat([train_label2, test_label], ignore_index = True)


# # train
# # -----------------------------
# # X
# # 把 train data 從 data_features2 抽出
# X_train = data_features3[0:46482]
# # 移除 key 跟 index 欄位
# X_train = X_train.drop(columns=['Key', 'Index',
#                                 'lat_norm', 'lng_norm',
#                                 '交易月份',
#                                 '移轉層次'])

# # y
# y_train = train_label2["price_per_ping"]
# # 轉一維
# y_train_array = y_train.values
# y_train_array = np.ravel(y_train_array)
# # -----------------------------


# test
# -----------------------------
## 把 test data 從 data_features2 抽出
X_test = data_features3[46482:]

## 移除 key 跟 index 欄位
# 跟其他
X_test = X_test.drop(columns=['Key', 'Index',
                                'lat_norm', 'lng_norm',
                                '交易月份',
                                '移轉層次'])
# # -----------------------------
# del data_features3, train_label2, y_train
# =============================================================================


# =============================================================================
X_train = data_features3
X_train = X_train.drop(columns=['Key', 'Index',
                                'lat_norm', 'lng_norm',
                                '交易月份',
                                '移轉層次'])


y_train = data_label["price_per_ping"]
# 轉一維
y_train_array = y_train.values
y_train_array = np.ravel(y_train_array)
# =============================================================================
del data_features3, data_label, test_label, train_label2, y_train




# SVM
from sklearn.svm import SVR
start_time = time.time()
# =============================================================================
svr = SVR()
svr.fit(X_train, y_train_array)

predictions = svr.predict(X_test)
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Program ran for {elapsed_time:.2f} seconds')
# 311.52 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions1.csv', index = True, encoding = "big5")
# =============================================================================



# RF
from sklearn.ensemble import RandomForestRegressor
start_time = time.time()
# =============================================================================
model = RandomForestRegressor()
model.fit(X_train, y_train_array)

predictions = model.predict(X_test)
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Program ran for {elapsed_time:.2f} seconds')
# 36.01 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions2.csv', index = True, encoding = "big5")
# =============================================================================



# 多項式回歸
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
start_time = time.time()
# =============================================================================
# 將訓練數據轉換為多項式特徵
poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)

# 建立
model = LinearRegression()
model.fit(X_train_poly, y_train_array)

# 將測試數據轉換為多項式特徵
X_test_poly = poly.transform(X_test)

# 預測
predictions = model.predict(X_test_poly)
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Program ran for {elapsed_time:.2f} seconds')
# 42.76 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions3.csv', index = True, encoding = "big5")
# =============================================================================



# XGboost
from xgboost import XGBRegressor
start_time = time.time()
# =============================================================================
model = XGBRegressor()
model.fit(X_train, y_train_array)

# 預測
predictions = model.predict(X_test)
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Program ran for {elapsed_time:.2f} seconds')
# 42.76 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions4.csv', index = True, encoding = "big5")
# =============================================================================

