import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# read data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding="big5")
y_train = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

# =============================================================================

# =============================================================================
# # df2 int
# 交易月份
# 土地移轉總面積(平方公尺)_log_2
# 建物現況格局-廳
# 建物現況格局-房
# 建物現況格局-衛
# 建物移轉總面積(平方公尺)_log
# 總樓層數
# num_of_bus_stations_in_100m	
# income_avg_log_2
# income_var_log_2
# low_use_electricity_log
# nearest_tarin_station_distance_log
# lat_norm
# lng_norm
# =============================================================================


# =============================================================================
# col = ['交易月份', 
#                                  '土地移轉總面積(平方公尺)_log_2',
#                                  '建物現況格局-廳',
#                                  '建物現況格局-房',
#                                  '建物現況格局-衛',
#                                  '建物移轉總面積(平方公尺)_log',
#                                  '總樓層數',
#                                  'num_of_bus_stations_in_100m',
#                                  'income_avg_log_2',
#                                  'income_var_log_2',
#                                  'low_use_electricity_log',
#                                  'nearest_tarin_station_distance_log',
#                                  'lat_norm',
#                                  'lng_norm']
# for c in col:

# #找Q1,Q3
#     q1, q3 = np.percentile(data_features2[c], [25, 75])
#     print(f"{data_features2[c]}'s Q1 is: {q1}, Q3 is: {q3}\n")

#     #上界及下界
#     above = q3 + 1.5 * (q3 - q1)
#     below = q1 - 1.5 * (q3 - q1)
#     print(f"{data_features2[c]}' s above is: {above}, Below is: {below}\n")
#     # 過濾
#     filtered_data = data_features2[(data_features2[c] > below) & (data_features2[c] < above)]
    
# =============================================================================


# =============================================================================
col2 = data_features3[['交易月份', 
                                 '土地移轉總面積(平方公尺)_log_2',
                                 '建物現況格局-廳',
                                 '建物現況格局-房',
                                 '建物現況格局-衛',
                                 '建物移轉總面積(平方公尺)_log',
                                 '總樓層數',
                                 'num_of_bus_stations_in_100m',
                                 'income_avg_log_2',
                                 'income_var_log_2',
                                 'low_use_electricity_log',
                                 'nearest_tarin_station_distance_log',
                                 'lat_norm',
                                 'lng_norm']]
# =============================================================================



# =============================================================================
# 過濾那些絕對值z_score 大於 3 的資料
from scipy import stats
z = np.abs(stats.zscore(col2))
filtered_data = col2[(z < 3).all(axis=1)]
# =============================================================================


# =============================================================================
filtered_data = pd.merge(
    filtered_data, data_features3, how="inner"
    # ['交易月份', 
    #                                  '土地移轉總面積(平方公尺)_log_2',
    #                                  '建物現況格局-廳',
    #                                  '建物現況格局-房',
    #                                  '建物現況格局-衛',
    #                                  '建物移轉總面積(平方公尺)_log',
    #                                  '總樓層數',
    #                                  'num_of_bus_stations_in_100m',
    #                                  'income_avg_log_2',
    #                                  'income_var_log_2',
    #                                  'low_use_electricity_log',
    #                                  'nearest_tarin_station_distance_log',
    #                                  'lat_norm',
    #                                  'lng_norm']
    )

filtered_data = filtered_data.drop_duplicates()

# 儲存
filtered_data.to_csv('data/filtered_data.csv', index = False, encoding = "big5")
# =============================================================================
