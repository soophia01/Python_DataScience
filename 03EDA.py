import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# read data
# =============================================================================
data_features2 = pd.read_csv("data/data_features2.csv",
                          encoding="big5")
train_features2 = data_features2[0:46482]
train_feature = pd.read_csv("data/train_feature.csv",
                          encoding="big5")

# 把label加回來
train_label_EDA = pd.read_csv("data/train_label_EDA.csv",
                          encoding="big5")

df1 = pd.merge(train_feature, train_label_EDA, on = "Index")
df2 = pd.merge(train_features2, train_label_EDA, on = "Index")
# =============================================================================
del data_features2, train_label_EDA, train_features2, train_feature


# 數值資料
# =============================================================================
# 原始
col_int1 = df1[['土地移轉總面積(平方公尺)',  '建物移轉總面積(平方公尺)', '都市土地使用分區', '鄉鎮市區', 'income_avg', 'income_var', 'location_type', 'low_use_electricity', 'nearest_tarin_station_distance', 'lat', 'lng', 'price_per_ping']]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
sns.pairplot(col_int1)


# 處理過
col_int2 = df2[['土地移轉總面積(平方公尺)_log_2',  '建物移轉總面積(平方公尺)_log', 'income_avg_log_2', 'income_var_log_2', 'location_type', 'low_use_electricity_log', 'nearest_tarin_station_distance_log', 'lat_norm', 'lng_norm', 'price_per_ping_log']]

sns.pairplot(col_int2)
# =============================================================================
# '交易月份', '土地', '建物', '車位', '建物現況格局-廳', '建物現況格局-房', '建物現況格局-衛', '總樓層數', 'num_of_bus_stations_in_100m', 
