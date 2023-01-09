import pandas as pd
import numpy as np
import time

# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# 移除 key 跟 index 欄位
data_features3 = data_features3.drop(columns=['Key', 'Index'])

# 移除 其他不想要的
# data_features3 = data_features3.drop(
#     columns=[
#             '交易月份', 
#             '交易標的',
#             '建物現況格局-隔間',
#             '建物移轉總面積(平方公尺)_log',
#             '移轉層次',
#             'nearest_tarin_station',
#             'low_use_electricity_log',
#             'nearest_tarin_station_distance_log',
#             'lat_norm',
#             'lng_norm',
#             ])


# 保留要的
## 相關係數
# data_features3 = data_features3[['nearest_tarin_station_distance_log',
#          '建物現況格局-房', '建物現況格局-廳',
#          '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', 
#          '總樓層數', '主要建材',
#          '鄉鎮市區', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地',
#          '都市土地使用分區',
#          'income_var_log_2', 'income_avg_log_2']]

## RF
# data_features3 = data_features3[['土地', '土地移轉總面積(平方公尺)_log_2', '建物移轉總面積(平方公尺)_log', '都市土地使用分區','num_of_bus_stations_in_100m', 'low_use_electricity_log']]

## 自己挑
data_features3 = data_features3[['土地', '土地移轉總面積(平方公尺)_log_2', '建物型態', '建物現況格局-廳', '建物現況格局-房', '總樓層數' ,'都市土地使用分區', '鄉鎮市區', 'num_of_bus_stations_in_100m', 'income_avg_log_2', 'income_var_log_2']]


# train
# -----------------------------
# X
# 把 train data 從 data_features2 抽出
X_train = data_features3[0:46482]

# y
y_train = train_label["price_per_ping"]
# 轉一維
y_train = y_train.values
y_train = np.ravel(y_train)
# -----------------------------


# test
# -----------------------------
## 把 test data 從 data_features2 抽出
X_test = data_features3[46482:]

# -----------------------------
del data_features3
# =============================================================================



# RF
from sklearn.ensemble import RandomForestRegressor
# =============================================================================
model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# =============================================================================
# Program ran for 85.16 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_RF.csv', index = True, encoding = "big5")
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
model.fit(X_train_poly, y_train)

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
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_poly.csv', index = True, encoding = "big5")
# =============================================================================



# XGboost
from xgboost import XGBRegressor
# =============================================================================
model = XGBRegressor()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
# =============================================================================
# Program ran for 4.25 seconds

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_XG.csv', index = True, encoding = "big5")
# =============================================================================



# SVM
from sklearn.svm import SVR
start_time = time.time()
# =============================================================================
# svr = SVR(C = 100, gamma = 0.1)
# svr.fit(X_train, y_train)

# predictions = svr.predict(X_test)
# =============================================================================
# Program ran for 1039.39 seconds

# 儲存
# # =============================================================================
# predictions = pd.DataFrame(predictions)
# predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
# predictions.index.name = "Index"
# predictions.to_csv('data/pred/predictions_SVM.csv', index = True, encoding = "big5")
# # =============================================================================

