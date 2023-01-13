import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

# df2 int
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

# 交易月份
fig = go.Figure()

# 土地移轉總面積(平方公尺)
# fig.add_trace(go.Box(
#     y=np.array(df1['土地移轉總面積(平方公尺)']),
#     name="土地移轉總面積(平方公尺)",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# 建物移轉總面積(平方公尺)_log
# fig.add_trace(go.Box(
#     y=np.array(df1["建物移轉總面積(平方公尺)"]),
#     name="建物移轉總面積(平方公尺)",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# 建物現況格局-廳
# 建物現況格局-房
# 建物現況格局-衛
# fig.add_trace(go.Box(
#     y=np.array(df2["建物現況格局-廳"]),
#     name="建物現況格局-廳",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# fig.add_trace(go.Box(
#     y=np.array(df2["建物現況格局-房"]),
#     name="建物現況格局-房",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# fig.add_trace(go.Box(
#     y=np.array(df2["建物現況格局-衛"]),
#     name="建物現況格局-衛",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# 總樓層數
# fig.add_trace(go.Box(
#     y=np.array(df2["總樓層數"]),
#     name="總樓層數",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# num_of_bus_stations_in_100m
# fig.add_trace(go.Box(
#     y=np.array(df2["num_of_bus_stations_in_100m"]),
#     name="num_of_bus_stations_in_100m",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# income_avg_log_2
# fig.add_trace(go.Box(
#     y=np.array(df2["income_avg_log_2"]),
#     name="income_avg_log_2",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# income_var_log_2
fig.add_trace(go.Box(
    y=np.array(df2["income_var_log_2"]),
    name="income_var_log_2",
    boxpoints='suspectedoutliers',  # only suspected outliers
    marker=dict(
        color='rgb(8,81,156)',
        outliercolor='rgba(219, 64, 82, 0.6)',
        line=dict(
            outliercolor='rgba(219, 64, 82, 0.6)',
            outlierwidth=2)),
    line_color='rgb(8,81,156)'
))


# low_use_electricity_log
# fig.add_trace(go.Box(
#     y=np.array(df2["low_use_electricity_log"]),
#     name="low_use_electricity_log",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# nearest_tarin_station_distance_log
# fig.add_trace(go.Box(
#     y=np.array(df2["nearest_tarin_station_distance_log"]),
#     name="nearest_tarin_station_distance_log",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# # lat_norm
# fig.add_trace(go.Box(
#     y=np.array(df2["lng_norm"]),
#     name="lat_norm",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))
# fig.add_trace(go.Box(
#     y=np.array(df1["lng"]),
#     name="lng",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

# lng_norm
# fig.add_trace(go.Box(
#     y=np.array(df2["lat_norm"]),
#     name="lat_norm",
#     boxpoints='suspectedoutliers',  # only suspected outliers
#     marker=dict(
#         color='rgb(8,81,156)',
#         outliercolor='rgba(219, 64, 82, 0.6)',
#         line=dict(
#             outliercolor='rgba(219, 64, 82, 0.6)',
#             outlierwidth=2)),
#     line_color='rgb(8,81,156)'
# ))

fig.update_layout(
    title_text="Box Plot",
     font=dict(
        family="Microsoft JhengHei",
        size=18,
    )
)
fig.show()

