import pandas as pd
import numpy as np
import time

# read train data
# =============================================================================
df = pd.read_csv("data/data_features4.csv",
                          encoding = "big5")

train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# 移除 key 跟 index 欄位
df = df.drop(columns=['Key', 'Index'])

# 移除 其他不想要的
# df = df.drop(
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
# df = df[['nearest_tarin_station_distance_log',
#          '建物現況格局-房', '建物現況格局-廳',
#          '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', 
#          '總樓層數', '主要建材',
#          '鄉鎮市區', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地',
#          '都市土地使用分區',
#          'income_var_log_2', 'income_avg_log_2']]

## RF
# df = df[['土地', '土地移轉總面積(平方公尺)_log_2', '建物移轉總面積(平方公尺)_log', '都市土地使用分區','num_of_bus_stations_in_100m', 'low_use_electricity_log']]

## 自己挑
# df = df[['土地', '土地移轉總面積(平方公尺)_log_2', '建物型態', '建物現況格局-廳', '建物現況格局-房', '總樓層數' ,'都市土地使用分區', '鄉鎮市區', 'num_of_bus_stations_in_100m', 'income_avg_log_2', 'income_var_log_2']]

## 鄉鎮ord_相關 data_features4
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地', '都市土地使用分區', 'income_var_log_2','income_avg_log_2', '鄉鎮市區']]

## 鄉鎮lab_相關 data_features3
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地_log_2', '都市土地使用分區', 'income_var_log_2','income_avg_log_2', '鄉鎮市區']]

#  鄉鎮lab_相關 data_features3 自己挑 的交集
# df = df[['nearest_tarin_station_distance_log', '都市土地使用分區',
#          '建物現況格局-房','建物現況格局-廳','總樓層數','土地移轉總面積(平方公尺)_log_2','建物型態', 'income_var_log_2','income_avg_log_2']]

# 自己挑2
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地_log_2', '都市土地使用分區', 'income_var_log_2','income_avg_log_2',
#          '鄉鎮市區',
#          '有無管理組織']]

# info
# ['主要建材', '主要用途', '交易標的', '土地_log_2', '土地', '建物_log_2', '建物', '建物現況格局-廳', '建物現況格局-隔間', '都市土地使用分區']

df = df[['土地_log_2', '土地移轉總面積(平方公尺)_log_2', '建物型態', '建物現況格局-廳', '建物現況格局-房', '建物現況格局-隔間', '總樓層數' ,'都市土地使用分區', '鄉鎮市區', 'num_of_bus_stations_in_100m', 'income_avg_log_2', 'income_var_log_2', '有無管理組織', '建物移轉總面積(平方公尺)_log']]

# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地_log_2', '都市土地使用分區', 'income_var_log_2','income_avg_log_2', '鄉鎮市區',
#          '有無管理組織']]


# df = df[['土地',  'num_of_bus_stations_in_100m',]]
# train
# -----------------------------
# X
# 把 train data 抽出
X_train = df[0:46482]

# y
y_train = train_label["price_per_ping"]
# 轉一維
y_train = y_train.values
y_train = np.ravel(y_train)
# -----------------------------

# test
# -----------------------------
## 把 test data 抽出
X_test = df[46482:]
# -----------------------------
del df
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


# knn
from sklearn.neighbors import KNeighborsRegressor
# =============================================================================
# 建立 KNN 回歸模型
knn = KNeighborsRegressor(n_neighbors = 50)

# 訓練模型
knn.fit(X_train, y_train)

# 預測測試資料
predictions = knn.predict(X_test)
# =============================================================================

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_knn.csv', index = True, encoding = "big5")
# =============================================================================



# 梯度提升決策樹(Gradient Boosting Decision Tree, GBDT)
from sklearn.ensemble import GradientBoostingRegressor
# =============================================================================
# 建立 GBDT 模型
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                 max_depth=1, random_state=42)
# 訓練模型
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
# =============================================================================

# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_GBDT.csv', index = True, encoding = "big5")
# =============================================================================




# # 線性回歸
# from sklearn.linear_model import LinearRegression
# # =============================================================================
# # 建立
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 預測
# predictions = model.predict(X_test)
# # =============================================================================
# # Program ran for 2.23 seconds

# # 儲存
# # =============================================================================
# predictions = pd.DataFrame(predictions)
# predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
# predictions.index.name = "Index"
# predictions.to_csv('data/pred/predictions_line.csv', index = True, encoding = "big5")
# # =============================================================================


# XGboost
from xgboost import XGBRegressor
# =============================================================================
model = XGBRegressor(
                    # colsample_bytree = 0.7,
                    #  learning_rate = 0.03,
                    #  max_depth = 6,
                    #  min_child_weight = 4,
                    #  n_estimators = 500,
                    #  nthread = 4,
                    #  objective = 'reg:squarederror',
                    #  # silent = 1,
                    #  subsample = 0.7
                     )
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
# from sklearn.svm import SVR
# start_time = time.time()
# # =============================================================================
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



# 多項式回歸
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# # =============================================================================
# # 將訓練數據轉換為多項式特徵
# poly = PolynomialFeatures(degree = 2)
# X_train_poly = poly.fit_transform(X_train)

# # 建立
# model = LinearRegression()
# model.fit(X_train_poly, y_train)

# # 將測試數據轉換為多項式特徵
# X_test_poly = poly.transform(X_test)

# # 預測
# predictions = model.predict(X_test_poly)
# # =============================================================================
# # Program ran for 2.23 seconds

# # 儲存
# # =============================================================================
# predictions = pd.DataFrame(predictions)
# predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
# predictions.index.name = "Index"
# predictions.to_csv('data/pred/predictions_poly.csv', index = True, encoding = "big5")
# # =============================================================================

