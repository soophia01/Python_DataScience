import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras



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


# 深度學習
# =============================================================================
# 創建一個神經網絡模型
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(keras.layers.Dense(units=1))

#編譯模型。在編譯時，你需要指定損失函數和優化器。
model.compile(loss='mean_squared_error', optimizer='adam')

# 訓練
model.fit(X_train, y_train_array, epochs = 10, batch_size = 32)

# 預測
predictions = model.predict(X_test)
# =============================================================================
