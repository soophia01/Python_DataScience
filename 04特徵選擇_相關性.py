import pandas as pd

# read train data
# =============================================================================
data_features1 = pd.read_csv("data/data_features.csv",
                          encoding = "big5")

data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")

y_train = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# 移除 key 跟 index 欄位
# data_features3 = data_features3.drop(columns=['Key', 'Index'])

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


# 把 train data 從 data_features3 抽出
X_train3 = data_features3[0:46482]
data_features3 = pd.merge(X_train3, y_train, on="Index")
data_features3 = data_features3.drop(columns=['Key', 'Index'])

# 把 train data 從 data_features 抽出
X_train1 = data_features1[0:46482]
data_features1 = pd.merge(X_train1, y_train, on="Index")
data_features1 = data_features1.drop(columns=['Index'])
# =============================================================================



# 相關係數
# 計算特徵與目標值的相關係數
correlations1 = data_features1.corr()['price_per_ping'].sort_values()
# 選擇相關係數較大的特徵
selected_features1 = correlations1[abs(correlations1) > 0.02].index
print(selected_features1)

# 計算特徵與目標值的相關係數
correlations3 = data_features3.corr()['price_per_ping'].sort_values()
# 選擇相關係數較大的特徵
selected_features3 = correlations3[abs(correlations3) > 0.02].index
print(selected_features3)


# -----------------------------
selected_features3 = pd.DataFrame(selected_features3)
# selected_features3.to_csv('data/pred/selected_features3_相關係數.csv', index = True, encoding = "big5")
del data_features1, data_features3, X_train1, X_train3, y_train

