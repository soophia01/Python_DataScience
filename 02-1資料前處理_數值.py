import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# read data
# =============================================================================
data_features = pd.read_csv("data/data_features.csv",
                          encoding="big5")
data_features = data_features.rename(columns = {data_features.columns[0]: "Key"})

train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")
# =============================================================================



# 處理缺失值: 數值欄位
# =============================================================================
# 平均: 'num_of_bus_stations_in_100m', 'income_avg', 'income_var', 'low_use_electricity'
# ---------------------------------------
null_columns_mean = ['num_of_bus_stations_in_100m', 'income_avg', 'income_var', 'low_use_electricity']

for column in null_columns_mean:
    mean = data_features[column].mean()
    data_features[column].fillna(mean, inplace=True)
    
# 也可以這樣寫: 
# data_features2[null_columns_mean] = data_features2[null_columns_mean].fillna(data_features2[null_columns_mean].mean())
# ---------------------------------------


# 眾數: '總樓層數', 'nearest_tarin_station_distance_log', 'lat_norm', 'lng_norm'
# ---------------------------------------
null_columns_mode = ['總樓層數', 'nearest_tarin_station_distance', 'lat', 'lng']

for column in null_columns_mode:
    mode = data_features[column].mode().values[0]
    data_features[column].fillna(mode, inplace = True)
    # 也可以這樣寫
    # data_features2[column].fillna(mode, inplace=True)
# ---------------------------------------
# =============================================================================



# 畫出數值欄位的 historgram
# =============================================================================
int_columns = ['土地', '建物', '車位', '土地移轉總面積(平方公尺)', '建物移轉總面積(平方公尺)', '總樓層數', 'num_of_bus_stations_in_100m', 'income_avg', 'income_var', 'low_use_electricity', 'nearest_tarin_station_distance', 'lat', 'lng']

for column in int_columns:
    data = pd.Series(data_features[column])
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 讓中文字體正確輸出
    sns.distplot(data)
    # plt.hist(data)
    plt.xlim(data.min(), data.max())
    plt.show()
    
    
# train_label
price_per_ping = pd.Series(train_label.iloc[:, 1])
sns.distplot(price_per_ping)
plt.xlim(price_per_ping.min(), price_per_ping.max())
plt.show()
# =============================================================================



# 對數縮放: '土地移轉總面積(平方公尺)', '建物移轉總面積(平方公尺)', 'nearest_tarin_station_distance', 'income_avg', 'income_var', 'low_use_electricity'
# =============================================================================
int_columns_log = ['土地移轉總面積(平方公尺)', '建物移轉總面積(平方公尺)', 'nearest_tarin_station_distance', 'income_avg', 'income_var', 'low_use_electricity']

for column in int_columns_log:
    data = pd.Series(data_features[column])
    data_log1p = np.log1p(data)
    # 把轉換後的資料取代回原本的df對應欄位位置
    col_index = data_features.columns.get_loc(column)
    # data_features = data_features.drop(columns = column)
    data_features.insert(loc = col_index, column = column+'_log', value = data_log1p)
    # 轉換後的圖表
    ## 檢查數值中是否包含 NaN 或無限大的數值，因為這些欄位裡面有缺失值還沒處理，會報錯(已先處理)
    has_nan = np.isnan(data_log1p).any()
    has_inf = np.isinf(data_log1p).any()
    # # 如果有 NaN 或無限大的數值，使用 nanmin 和 nanmax 來獲取數值中非 NaN 的最大值和最小值
    # if has_nan or has_inf:
    #     min_val = np.nanmin(data_log1p)
    #     max_val = np.nanmax(data_log1p)
    # ## 否則使用 min 和 max 來獲取數值中的最大值和最小值
    # else:
    #     min_val = data_log1p.min()
    #     max_val = data_log1p.max()
    sns.distplot(pd.Series(data_features[column+'_log']))
    plt.xlim(data_log1p.min(), data_log1p.max())
    plt.show()
    
    
# price_per_ping
# ------------------------
data_log1p = np.log1p(price_per_ping)
train_label.insert(loc = 2, column = 'price_per_ping_log', value = data_log1p)

                   
sns.distplot(data_log1p)
plt.xlim(data_log1p.min(), data_log1p.max())
plt.show()



# data_log1p = np.log1p(data_log1p)
# train_label.insert(loc = 2, column = 'price_per_ping_log_2', value = data_log1p)

                   
# sns.distplot(data_log1p)
# plt.xlim(data_log1p.min(), data_log1p.max())
# plt.show()
# ------------------------
# =============================================================================


# 再取一次
# =============================================================================
int_columns_log = ['土地移轉總面積(平方公尺)_log', 'income_avg_log', 'income_var_log']

for column in int_columns_log:
    data = pd.Series(data_features[column])
    data_log1p = np.log1p(data)
    # 把轉換後的資料取代回原本的df對應欄位位置
    col_index = data_features.columns.get_loc(column)
    data_features = data_features.drop(columns = column)
    data_features.insert(loc = col_index, column = column+'_2', value = data_log1p)
    # 轉換後的圖表
    ## 檢查數值中是否包含 NaN 或無限大的數值，因為這些欄位裡面有缺失值還沒處理，會報錯(已先處理)
    has_nan = np.isnan(data_log1p).any()
    has_inf = np.isinf(data_log1p).any()
    sns.distplot(pd.Series(data_features[column+'_2']))
    plt.xlim(data_log1p.min(), data_log1p.max())
    plt.show()
# =============================================================================



# 移除兩次log之後不需要的多於欄位
# =============================================================================
int_columns_log = ['土地移轉總面積(平方公尺)', '建物移轉總面積(平方公尺)', 'nearest_tarin_station_distance', 'income_avg', 'income_var', 'low_use_electricity']
for column in int_columns_log:
    data_features = data_features.drop(columns = column)
# =============================================================================



# 標準化: 'low_use_electricity', 'nearest_tarin_station_distance',  'income_avg', 'income_var'
# =============================================================================
# int_columns_sd = ['low_use_electricity', 'nearest_tarin_station_distance',  'income_avg', 'income_var']

# for column in int_columns_sd:
#     data = pd.Series(data_features[column])
#     # 將資料轉換成 numpy array，並進行標準化
#     data = data.values.reshape(-1, 1)
#     scaler = StandardScaler()
#     data_sd = scaler.fit_transform(data)
#     # 將標準化後的資料取代回原本的df對應欄位位置
#     col_index = data_features.columns.get_loc(column)
#     # data_features = data_features.drop(columns = column)
#     data_features.insert(loc = col_index, column = column+'_sd', value = data_sd)
#     # 轉換後的圖表
#     sns.distplot(pd.Series(data_features[column+'_sd']))
#     plt.xlim(data_sd.min(), data_sd.max())
#     plt.show()

# price_per_ping
# ------------------------
price_per_ping = data_log1p.values.reshape(-1, 1)
scaler = StandardScaler()
price_per_ping_sd = scaler.fit_transform(price_per_ping)
train_label.insert(loc = 2, column = 'price_per_ping_log_sd', value = price_per_ping_sd)

                   
sns.distplot(price_per_ping_sd)
plt.xlim(price_per_ping_sd.min(), price_per_ping_sd.max())
plt.show()
# ------------------------
# =============================================================================



# Min-max scaling 將資料縮放到 0 到 1 之間，並不會改變數值的相對大小關係。
# normalization 則將資料轉換到向量長度為 1 的分布，這有助於突顯數值之間的差異。
# normalization: 'lat', 'lng'
# =============================================================================
int_columns_norm = ['lat', 'lng']


for column in int_columns_norm:
    # 取出資料
    data = pd.Series(data_features[column])
    # 進行 normalization
    data_norm = (data - data.mean()) / data.std()
    # 把轉換後的資料取代回原本的df對應欄位位置
    col_index = data_features.columns.get_loc(column)
    data_features = data_features.drop(columns = column)
    data_features.insert(loc = col_index, column = column+'_norm', value = data_norm)
    # 轉換後的圖表
    sns.distplot(pd.Series(data_features[column+'_norm']))
    plt.show()
# =============================================================================



# 儲存
# =============================================================================
data_features.to_csv('data/data_features2.csv', index = False, encoding = "big5")


train_label.to_csv('data/train_label2.csv', index = False, encoding = "big5")
# =============================================================================
