import pandas as pd

# read train data
# =============================================================================
data_features_含數值sd = pd.read_csv("data/data_features_含數值sd.csv",
                          encoding = "big5")
y_train = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

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


# 把 train data 抽出
X = data_features_含數值sd[0:46482]
X = pd.merge(X, y_train, on="Index")
X = X.drop(columns=['Key', 'Index'])


# =============================================================================



# 相關係數
# 計算特徵與目標值的相關係數
correlations = X.corr()['price_per_ping'].sort_values()
# 選擇相關係數較大的特徵
selected_features_sd= correlations[abs(correlations) >= 0.02].index
print(selected_features_sd)

# # 計算特徵與目標值的相關係數
# correlations3 = data_features3.corr()['price_per_ping'].sort_values()
# # 選擇相關係數較大的特徵
# selected_features3 = correlations3[abs(correlations3) > 0.02].index
# print(selected_features3)


# -----------------------------
# selected_features3 = pd.DataFrame(selected_features3)
# selected_features3.to_csv('data/pred/selected_features3_相關係數.csv', index = True, encoding = "big5")
# del data_features4, data_features3, X_train1, X_train3, y_train

