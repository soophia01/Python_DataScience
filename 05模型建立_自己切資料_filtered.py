import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error


# read train data
# =============================================================================
filtered_data = pd.read_csv("data/filtered_data.csv",
                          encoding = "big5")
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

# x
X_train_filtered = filtered_data[filtered_data["Key"] < 46482 ]
X_train_filtered = X_train_filtered.drop(columns=['Key', 'Index'])

# y
y_train = X_train_filtered[["price_per_ping"]]
# 轉一維
y_train = y_train.values
y_train = np.ravel(y_train)
# -----------------------------

## outlier
X_train_filtered = X_train_filtered[['lat_norm', '建物現況格局-房', '建物現況格局-廳', '交易標的', '建物現況格局-衛',
       '建物移轉總面積(平方公尺)_log', 'nearest_tarin_station_distance_log', '建物型態',
       'num_of_bus_stations_in_100m', 'income_var_log_2', 'lng_norm',
       'income_avg_log_2']]

# X_train_filtered = X_train_filtered.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X_train_filtered, y_train, test_size = 0.2, random_state = 42)

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
model = XGBRegressor(
    # colsample_bytree=0.7,
    # learning_rate=0.03,
    # max_depth=6,
    # min_child_weight=4,
    # n_estimators=500,
    # nthread=4,
    # objective='reg:linear',
    # silent=1,
    # subsample=0.7
)
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
