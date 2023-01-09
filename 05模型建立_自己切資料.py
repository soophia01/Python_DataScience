import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error



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
# X_test = data_features3[46482:]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# -----------------------------
del data_features3, train_label
# =============================================================================



###############################################################################



# RF
from sklearn.ensemble import RandomForestRegressor
# =============================================================================
model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# =============================================================================
# Program ran for 85.16 seconds

# 評估
# =============================================================================
# 計算平均絕對誤差
mae = mean_absolute_error(y_test, predictions)

# 計算平均平方誤差
mse = mean_squared_error(y_test, predictions)

# 計算平方根平均平方誤差
rmse = np.sqrt(mean_squared_error(y_test, predictions))

evalu_RF = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# 輸出資料表
print(evalu_RF)
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

# 評估
# =============================================================================
# 計算平均絕對誤差
mae = mean_absolute_error(y_test, predictions)

# 計算平均平方誤差
mse = mean_squared_error(y_test, predictions)

# 計算平方根平均平方誤差
rmse = np.sqrt(mean_squared_error(y_test, predictions))

evalu_poly = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# 輸出資料表
print(evalu_RF)
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

# 評估
# =============================================================================
# 計算平均絕對誤差
mae = mean_absolute_error(y_test, predictions)

# 計算平均平方誤差
mse = mean_squared_error(y_test, predictions)

# 計算平方根平均平方誤差
rmse = np.sqrt(mean_squared_error(y_test, predictions))

evalu_XG = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# 輸出資料表
print(evalu_XG)
# =============================================================================



# # SVM
# from sklearn.svm import SVR
# start_time = time.time()
# # =============================================================================
# svr = SVR(C = 100, gamma = 0.1)
# svr.fit(X_train, y_train)

# predictions = svr.predict(X_test)
# # =============================================================================
# # Program ran for 1039.39 seconds

# # SVM 參數調整
# from sklearn.model_selection import GridSearchCV # 自動對一組參數進行評估，並找到最佳的參數組合
# # =============================================================================
# # param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}

# # 使用 GridSearchCV 进行参数调优
# # grid_search = GridSearchCV(svr, param_grid, cv = 5)
# # grid_search.fit(X_train, y_train)

# # 输出最优参数
# # print(f'最優參數: {grid_search.best_params_}')
# # 最優參數: {'C': 100, 'gamma': 0.1}
# # =============================================================================
# # end_time = time.time()
# # elapsed_time = end_time - start_time
# # print(f'參數調整: {elapsed_time:.2f} seconds')

# # 評估
# # =============================================================================
# # 計算平均絕對誤差
# mae = mean_absolute_error(y_test, predictions)

# # 計算平均平方誤差
# mse = mean_squared_error(y_test, predictions)

# # 計算平方根平均平方誤差
# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# evalu_XG = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# # 輸出資料表
# print(evalu_XG)
# # =============================================================================
