import pandas as pd
from sklearn.preprocessing import LabelEncoder

# read data
# =============================================================================
data_features2 = pd.read_csv("data/data_features2.csv",
                          encoding="big5")
# =============================================================================


# table
# =============================================================================
# columns = data_features2['建物型態']
# for var in columns:
#     print(data_features2[var].value_counts())
# =============================================================================




# 處理缺失值: 類別欄位
# null_data_features2 = data_features2.isnull().sum()
# =============================================================================

# 出現次數最多的類別: '主要建材', '主要用途', '移轉層次'
# ---------------------------------------
null_columns = data_features2[['主要建材', '主要用途', '移轉層次']]

for column in null_columns:
    # 統計每個類別的出現次數
    counts = data_features2[column].value_counts()
    # 找出出現次數最多的類別
    most_common = counts.index[0]
    # 補
    data_features2[column].fillna(most_common, inplace = True)
# ---------------------------------------


# 新增"未知"欄位: '都市土地使用分區'
# ---------------------------------------
# 對缺失值補"unknown"
data_features2['都市土地使用分區'].fillna('Unknown', inplace = True)
# ---------------------------------------
# =============================================================================



# 儲存
# =============================================================================
data_features2.to_csv('data/data_features2.csv', index = False, encoding = "big5")
# =============================================================================
data_features3 = data_features2


# one hot encoding
# =============================================================================
data_features3 = pd.get_dummies(data_features3, columns=['都市土地使用分區'])
# =============================================================================



# binary encoding
# =============================================================================
data_features3['建物現況格局-隔間'].replace(['有', '無'], [1,0], inplace = True)
data_features3['有無管理組織'].replace(['有', '無'], [1,0], inplace = True)
# =============================================================================



# label encoding: '主要建材','主要用途','交易標的', '建物型態','移轉層次','鄉鎮市區','location_type','nearest_tarin_station'
# =============================================================================
le = LabelEncoder()
labeled_data = data_features3[['主要建材',
                              '主要用途',
                              '交易標的',
                              '建物型態',
                              '移轉層次',
                              '鄉鎮市區',
                              'location_type',
                              'nearest_tarin_station']]
labeled_data = labeled_data.apply(le.fit_transform)

# 把轉換後的欄位存回原本的df
labeled_data_columnName = labeled_data.columns.tolist()

## 把相同欄位名稱用轉換後的資料取代
def replace_column_values(A, B, column_name):
  A[column_name] = B[column_name]
replace_column_values(data_features3, labeled_data, labeled_data_columnName)
# =============================================================================



# OrdinalEncoder
# =============================================================================
# 主要建材

# mapping = {'鋼筋混凝土造': 1,
#            '鋼骨鋼筋混凝土造': 2,
#            '鋼骨混凝土造': 3,
#            '預力混凝土造': 4,
#            '鋼筋混凝土加強磚造': 5,
#            '加強磚造': 6,
#            '壁式預鑄鋼筋混凝土造': 7,
#            '磚造': 8,
#            '土磚石混合造': 9,
#            '木造': 10,
#            '石造': 11,
#            '土木造': 12,
#            '土造': 13,
#            '竹造': 14,
#            '鐵造': 15,
#            '見使用執照': 16,
#            '見其他登記事項': 17,
#            # NaN: 0
#            }
# 
# data_features3["主要建材"] = data_features2["主要建材"] .map(mapping)

# =============================================================================



# 儲存
# =============================================================================
data_features3.to_csv('data/data_features3.csv', index = False, encoding = "big5")
# =============================================================================
