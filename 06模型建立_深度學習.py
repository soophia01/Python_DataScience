import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# 移除 key 跟 index 欄位
data_features3 = data_features3.drop(columns=['Key', 'Index'])

# 移除 其他不想要的
data_features3 = data_features3.drop(
    columns=[
            '交易月份', 
            '交易標的',
            '建物現況格局-隔間',
            '建物移轉總面積(平方公尺)_log',
            '移轉層次',
            'nearest_tarin_station',
            'low_use_electricity_log',
            'nearest_tarin_station_distance_log',
            'lat_norm',
            'lng_norm',
            ])
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

# =============================================================================


# 深度學習
# =============================================================================
# 創建一個神經網絡模型
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(keras.layers.Dense(units=1))

#編譯模型。在編譯時，你需要指定損失函數和優化器。
model.compile(loss='mean_squared_error', optimizer='adam')

# 訓練
# model.fit(X_train, y_train, epochs = 10, batch_size = 32)
model.fit(X_train, y_train, epochs = 50, batch_size = 100)

# 預測
predictions = model.predict(X_test)
# =============================================================================


# 儲存
# =============================================================================
predictions = pd.DataFrame(predictions)
predictions = predictions.rename(columns = {predictions.columns[0]: "price_per_ping"})
predictions.index.name = "Index"
predictions.to_csv('data/pred/predictions_DL2.csv', index = True, encoding = "big5")
# =============================================================================

