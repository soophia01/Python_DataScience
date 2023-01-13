import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error



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
df = df[['nearest_tarin_station_distance_log',
          '建物現況格局-房', '建物現況格局-廳',
          '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', 
          '總樓層數', '主要建材',
          '鄉鎮市區', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地',
          '都市土地使用分區',
          'income_var_log_2', 'income_avg_log_2']]

## RF
# df = df[['土地', '土地移轉總面積(平方公尺)_log_2', '建物移轉總面積(平方公尺)_log', '都市土地使用分區','num_of_bus_stations_in_100m', 'low_use_electricity_log']]

## 自己挑
# df = df[['土地', '土地移轉總面積(平方公尺)_log_2', '建物型態', '建物現況格局-廳', '建物現況格局-房', '總樓層數' ,'都市土地使用分區', '鄉鎮市區', 'num_of_bus_stations_in_100m', 'income_avg_log_2', 'income_var_log_2']]

## 鄉鎮ord_相關 data_features4
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地', '都市土地使用分區', 'income_var_log_2','income_avg_log_2', '鄉鎮市區']]

## 鄉鎮lab_相關 data_features3
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-隔間', 'lat_norm', 'nearest_tarin_station', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地_log_2', '都市土地使用分區', 'income_var_log_2','income_avg_log_2', '鄉鎮市區']]


## 鄉鎮lab_相關 data_features3  , '建物現況格局-隔間'X
# df = df[['nearest_tarin_station_distance_log', '建物現況格局-房', '建物現況格局-廳', 'lat_norm', '總樓層數', '主要建材', '土地移轉總面積(平方公尺)_log_2', '建物型態', '土地_log_2', '都市土地使用分區', 'income_var_log_2','income_avg_log_2',
#           '鄉鎮市區',
#           'nearest_tarin_station', 
#          '有無管理組織']]

## info gain
# df = df[['主要建材', '主要用途', '交易標的', '土地_log_2', '建物_log_2', '建物現況格局-廳', '建物現況格局-衛', '建物現況格局-隔間', '都市土地使用分區', 'location_type']]

# df = df[['土地', '土地移轉總面積(平方公尺)_log_2', '建物型態', '建物現況格局-廳', '建物現況格局-房', '建物現況格局-隔間', '總樓層數' ,'都市土地使用分區', '鄉鎮市區', 'num_of_bus_stations_in_100m', 'income_avg_log_2', 'income_var_log_2','有無管理組織']]

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
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# -----------------------------
del df, train_label
# =============================================================================



###############################################################################



# RF
# =============================================================================
# 隨機森林（Random Forest）常見的參數包括：
# n_estimators：整数值，表示決策樹的數量。
# criterion：字符串值，表示計算節點的資訊增益的方式。可以是 "gini" 或 "entropy"。
# max_depth：整数值，表示決策樹的最大深度。
# min_samples_split：整数值或浮點数值，表示在建立子節點時所需的最小樣本數。
# min_samples_leaf：整数值或浮點数值，表示樹中葉子節點所需的最小樣本數。
# RandomForestClassifier(
#     n_estimators=100, 
#     criterion='gini', 
#     max_depth=None, 
#     min_samples_split=2, 
#     min_samples_leaf=1, 
#     min_weight_fraction_leaf=0.0, 
#     max_features='auto', 
#     max_leaf_nodes=None, 
#     min_impurity_decrease=0.0, 
#     min_impurity_split=None, 
#     bootstrap=True, 
#     oob_score=False, 
#     n_jobs=None, 
#     random_state=None, 
#     verbose=0, 
#     warm_start=False, 
#     class_weight=None,
# )
# =============================================================================
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
print("evalu_RF\n")
print(evalu_RF)
# =============================================================================



# # KNN
# # 交叉驗證
# # =============================================================================
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import cross_val_score
# knn = KNeighborsRegressor()
# scores = cross_val_score(knn, X_train, y_train, cv = 10)

# print("Cross Validation scores: ", scores)
# print("Average score: ", scores.mean())

# # =============================================================================

# # =============================================================================
# # 建立 KNN 回歸模型
# knn = KNeighborsRegressor(n_neighbors = 50)

# # 訓練模型
# knn.fit(X_train, y_train)

# # 預測測試資料
# predictions = knn.predict(X_test)
# # =============================================================================

# # 評估
# # =============================================================================
# # 計算平均絕對誤差
# mae = mean_absolute_error(y_test, predictions)

# # 計算平均平方誤差
# mse = mean_squared_error(y_test, predictions)

# # 計算平方根平均平方誤差
# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# evalu_knn = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# # 輸出資料表
# print("KNN\n")
# print(evalu_knn)
# # =============================================================================




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
print("evalu_poly\n")
print(evalu_poly)
# =============================================================================



# XGboost
# =============================================================================
# XGBoost 常見的參數包括：
# learning_rate：浮點数值，表示模型中使用的學習率。
# n_estimators：整数值，表示弱學習器的數量。
# max_depth：整数值，表示決策樹的最大深度。
# gamma：浮點数值，表示用於控制是否繼續分裂的參數。
# subsample：浮點数值，表示用於訓練模型的子樣本的大小。
# colsample_bytree：浮點数值，表示用於訓練每棵決策樹的特徵的子樣本的大小。
# XGBClassifier(
#     booster='gbtree', 
#     base_score=0.5, 
#     booster='gbtree', 
#     colsample_bylevel=1, 
#     colsample_bynode=1, 
#     colsample_bytree=1, 
#     gamma=0, 
#     learning_rate=0.1, 
#     max_delta_step=0, 
#     max_depth=3, 
#     min_child_weight=1, 
#     missing=None, 
#     n_estimators=100, 
#     n_jobs=1, 
#     nthread=None, 
#     objective='binary:logistic', 
#     random_state=0, 
#     reg_alpha=0, 
#     reg_lambda=1, 
#     scale_pos_weight=1, 
#     seed=None, 
#     silent=None, 
#     subsample=1, 
#     verbosity=1
# )

# =============================================================================
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

## 參數
# from sklearn.model_selection import GridSearchCV
# # Various hyper-parameters to tune
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], #so called `eta` value
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]}

# xgb_grid = GridSearchCV(model,
#                         parameters,
#                         cv = 2,
#                         n_jobs = 5,
#                         verbose=True)

# xgb_grid.fit(X_train, y_train)

# print(xgb_grid.best_score_) # -0.41794488813067093
# print(xgb_grid.best_params_)
# {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 500, 'nthread': 4, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}

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
print("evalu_XG\n")
print(evalu_XG)
# =============================================================================



# SVM
from sklearn.svm import SVR
# start_time = time.time()
# =============================================================================
svr = SVR(C = 100, gamma = 0.1)
svr.fit(X_train, y_train)

predictions = svr.predict(X_test)
# =============================================================================
# Program ran for 1039.39 seconds

# SVM 參數調整
from sklearn.model_selection import GridSearchCV # 自動對一組參數進行評估，並找到最佳的參數組合
# =============================================================================
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}

# 使用 GridSearchCV 进行参数调优
# grid_search = GridSearchCV(svr, param_grid, cv = 5)
# grid_search.fit(X_train, y_train)

# 输出最优参数
# print(f'最優參數: {grid_search.best_params_}')
# 最優參數: {'C': 100, 'gamma': 0.1}
# =============================================================================
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f'參數調整: {elapsed_time:.2f} seconds')

# 評估
# =============================================================================
# 計算平均絕對誤差
mae = mean_absolute_error(y_test, predictions)

# 計算平均平方誤差
mse = mean_squared_error(y_test, predictions)

# 計算平方根平均平方誤差
rmse = np.sqrt(mean_squared_error(y_test, predictions))

evalu_SVM = pd.DataFrame({'mae': mae, 'mse': mse, 'rmse': rmse}, index=[0])

# 輸出資料表
print("evalu_SVM\n")
print(evalu_SVM)
# =============================================================================
