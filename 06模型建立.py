import pandas as pd
import numpy as np
import time


# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

train_label2 = pd.read_csv("data/train_label2.csv",
                          encoding = "big5")


# train
# -----------------------------
# X
# 把 train data 從 data_features2 抽出
X_train = data_features3[0:46482]
# 移除 key 跟 index 欄位
X_train = X_train.drop(columns=['Key', 'Index',
                                'lat_norm', 'lng_norm',
                                '交易月份',
                                '移轉層次'])

# y
y_train = train_label2["price_per_ping"]
# 轉一維
y_train_array = y_train.values
y_train_array = np.ravel(y_train_array)
# -----------------------------


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
del data_features3, train_label2, y_train
# =============================================================================


# SVM
from sklearn.svm import SVR
start_time = time.time()
# =============================================================================
svr = SVR()
svr.fit(X_train, y_train_array)

predictions = svr.predict(X_test)
# =============================================================================
# Program ran for 1039.39 seconds


# SVM 參數調整
from sklearn.model_selection import GridSearchCV # 自動對一組參數進行評估，並找到最佳的參數組合
# =============================================================================
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}

# 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(svr, param_grid, cv = 5)
grid_search.fit(X_train, y_train_array)

# 输出最优参数
print(f'最優參數: {grid_search.best_params_:.2f}')
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f'參數調整: {elapsed_time:.2f} seconds')

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions1.csv', index = True, encoding = "big5")
# =============================================================================



# RF
from sklearn.ensemble import RandomForestRegressor
# =============================================================================
model = RandomForestRegressor()
model.fit(X_train, y_train_array)

predictions = model.predict(X_test)
# =============================================================================
# Program ran for 85.16 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions_RF.csv', index = True, encoding = "big5")
# =============================================================================



# 多項式回歸
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
# Program ran for 2.23 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions_poly.csv', index = True, encoding = "big5")
# =============================================================================



# XGboost
from xgboost import XGBRegressor
# =============================================================================
model = XGBRegressor()
model.fit(X_train, y_train_array)

# 預測
predictions = model.predict(X_test)
# =============================================================================
# Program ran for 4.25 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.to_csv('data/predictions_XG.csv', index = True, encoding = "big5")
# =============================================================================

