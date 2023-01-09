import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# 移除 key 跟 index 欄位
data_features3 = data_features3.drop(columns=['Key', 'Index'])


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
# 建立隨機森林模型
model = ExtraTreesRegressor()

# 訓練模型
model.fit(X_train, y_train)

# 取得各個特徵的重要性
importances = model.feature_importances_

# 選擇重要性較高的特徵
selected_features = X_train.columns[importances > 0.05]

print(selected_features)

# Index(['土地', '土地移轉總面積(平方公尺)_log_2', '建物移轉總面積(平方公尺)_log', '都市土地使用分區','num_of_bus_stations_in_100m', 'low_use_electricity_log'])