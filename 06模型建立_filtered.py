import pandas as pd
import numpy as np
import time

# read train data
# =============================================================================
filtered_data = pd.read_csv("data/filtered_data.csv",
                          encoding = "big5")
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

## outlier
filtered_data = filtered_data[['Key', 'lat_norm', '建物現況格局-房', '建物現況格局-廳', '交易標的', '建物現況格局-衛',
       '建物移轉總面積(平方公尺)_log', 'nearest_tarin_station_distance_log', '建物型態',
       'num_of_bus_stations_in_100m', 'income_var_log_2', 'lng_norm',
       'income_avg_log_2', 'price_per_ping']]

## outlier
data_features3 = data_features3[['Key', 'lat_norm', '建物現況格局-房', '建物現況格局-廳', '交易標的', '建物現況格局-衛',
       '建物移轉總面積(平方公尺)_log', 'nearest_tarin_station_distance_log', '建物型態',
       'num_of_bus_stations_in_100m', 'income_var_log_2', 'lng_norm',
       'income_avg_log_2', 'price_per_ping']]

# x
X_train = filtered_data[filtered_data["Key"] < 46482 ]
X_test = data_features3[data_features3["Key"] >= 46482 ]
# y
y_train = X_train[["price_per_ping"]]

X_train = X_train.drop(columns=['price_per_ping'])
X_test = X_test.drop(columns=['price_per_ping'])



# 轉一維
y_train = y_train.values
y_train = np.ravel(y_train)

# -----------------------------

#  =============================================================================

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
model = XGBRegressor(colsample_bytree = 0.7,
                     learning_rate = 0.03,
                     max_depth = 6,
                     min_child_weight = 4,
                     n_estimators = 500,
                     nthread = 4,
                     objective = 'reg:linear',
                     silent = 1,
                     subsample = 0.7)
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

