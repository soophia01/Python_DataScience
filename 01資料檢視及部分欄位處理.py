import pandas as pd
import re

# read data
# =============================================================================
# train data
train_feature = pd.read_csv("data/train_feature.csv",
                          encoding = "big5",
                           # dtype = object
                          ) 
                          
# test data
test_feature = pd.read_csv("data/test_feature.csv",
                          encoding = "big5")

# 合併所有資料_無label(price_per_ping)欄位
data_features = pd.concat([train_feature, test_feature], ignore_index = True)
# data_features = pd.read_csv("data/data_features.csv",
#                            encoding = "big5",
#                           dtype = object)
# =============================================================================



# 檢查 train, test 缺失值數量
# =============================================================================
# null_train = train_feature.isnull().sum()
# null_test = test_feature.isnull().sum()
# null_data_features = data_features.isnull().sum()
# =============================================================================
# 一起檢查
null_data_features = data_features.isnull().sum()




# 移除欄位_1 (不重要 / 缺值太多)
# =============================================================================
data_features = data_features.drop(columns = ["備註", '土地區段位置/建物區段門牌', "建築完成年月", "編號", "車位移轉總面積(平方公尺)", "車位類別", "非都市土地使用分區", "非都市土地使用編定"])
# =============================================================================



# 查看各欄位組成
## 類似 R table
# =============================================================================
# train_feature_column_names = list(train_feature.columns)[10:37]
# for var in train_feature_column_names:
#     print(train_feature[var].value_counts())
# =============================================================================




# 把數值型態的欄位(但實際資料型態為類別)做處理
# =============================================================================

# "%" 轉為 float
# -----------------------------------------
data_features['low_use_electricity'] = data_features['low_use_electricity'].str.replace('%', '').astype(float) / 100
# -----------------------------------------


# 欄位字串處理
# -----------------------------------------
# 移除"層"
data_features['總樓層數'] = data_features['總樓層數'].replace('層', '', regex = True)

# 把中文數字轉阿拉伯數字
data_features['總樓層數'] = data_features['總樓層數'].replace( ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十', '二十一', '二十二', '二十三', '二十四', '二十五', '二十六', '二十七', '二十八', '二十九', '三十', '三十一', '三十三', '三十五', '三十八', '四十二'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 38, 42], regex = True)
# -----------------------------------------



# 特別處理
# =============================================================================
# 交易筆棟數
# ---------------------------------------------

# 取得目前欄位位址
col_index = data_features.columns.get_loc('交易筆棟數')

# 新增 '土地', '建物', '車位' 這三個欄位
new_column = ['土地', '建物', '車位']
for column in new_column:
    data_features.insert(loc = col_index, column = column, value = None)
    col_index += 1


# 把'交易筆棟數' 裡面的值分別放進 new_column 裡面
data_features['交易筆棟數'] = data_features['交易筆棟數'].astype(str)

# 建立空列表，用來存儲每筆資料的結果
results = []

# 對每筆資料進行操作
for s in data_features['交易筆棟數']:
    numbers = re.findall(r'\d+', s)
    results.append(numbers)

# 将结果转换成 DataFrame
df = pd.DataFrame(results, columns=['土地', '建物', '車位'])

# 把相同欄位名稱用轉換後的資料取代
def replace_column_values(A, B, column_name):
  A[column_name] = B[column_name]
  
replace_column_values(data_features, df, new_column)

# 填完值後，刪除 '交易筆棟數' 欄位
data_features = data_features.drop(columns=['交易筆棟數'])
# ---------------------------------------------
# =============================================================================



# 時間
# -----------------------------------------
# 交易年月日，無缺值
## 把 "交易年月日" 的 00:00:00 去掉
data_features['交易年月日'] = data_features['交易年月日'].replace('00:00:00', '', regex = True)

## 取出月份
### 取"交易年月日" 的第5~6位 => 月份
data_features['交易年月日'] = [date_string[5:7] for date_string in data_features['交易年月日']]

### 把"交易年月日"改成"交易月份"
data_features.rename(columns = {'交易年月日': '交易月份'}, inplace = True)

### 欄位資料型態轉成數值
# data_features['交易月份'] = pd.to_datetime(data_features['交易月份'], format='%m')
data_features["交易月份"] = data_features["交易月份"].astype(int)
# -----------------------------------------

# 建築完成年月，最後決定移除，資料太髒

# 有缺值，且資料長度從 1~5 都有(刪除這些長度不符的資料)
## 補為 000000
# data_features['建築完成年月'].fillna("000000", inplace = True)
# data_features["建築完成年月"] = data_features["建築完成年月"].astype(int)
# data_features['建築完成年月'] = data_features['建築完成年月'].apply(str)

## 檢查欄位各值長度
# ------------------
# length_2_values = []
# length_3_values = []
# length_4_values = []
# length_5_values = []

# for key, value in data_features['建築完成年月'].items():
#     if len(str(value)) == 2:
#         length_2_values.append(key)
#     elif len(str(value)) == 3:
#         length_3_values.append(key)
#     elif len(str(value)) == 4:
#         length_4_values.append(key)
#     elif len(str(value)) == 5:
#         length_5_values.append(key)

# print(f"長度為 2 的 key 有 {length_2_values}")
# print(f"長度為 3 的 key 有 {length_3_values}")
# print(f"長度為 4 的 key 有 {length_4_values}")
# print(f"長度為 5 的 key 有 {length_5_values}")

########### result ###########
# 有 4 個欄位值的長度為 2: [9172, 37739, 44058, 55025]
# 有 1 個欄位值的長度為 3: [21368]
# 有 8 個欄位值的長度為 4: [10918, 14891, 20110, 25579, 28175, 29502, 36076, 45502]
# 有 6 個欄位值的長度為 5: [21121, 36358, 53319, 58892, 62922, 65531]
# ------------------

## 取出月份
### 取"建築完成年月" 的到數第 3~4位 => 月份
# data_features['建築完成年月'] = [date_string[-4:-2] for date_string in data_features['建築完成年月']]

# ########## test
# data = data_features["建築完成年月"]
# # 取出倒數第 3~4 位
# result = data.str[-4:-2]
# #############

# ### 把"建築完成年月"改成"建築完成月份"
# data_features.rename(columns = {'建築完成年月': '建築完成月份'}, inplace = True)
# ### 欄位資料型態轉成數值
# # data_features['建築完成月份'] = pd.to_datetime(data_features['建築完成月份'], format='%m')
# data_features['建築完成月份'] = data_features['建築完成月份'].apply(pd.to_numeric)
# -----------------------------------------
# =============================================================================



# 存檔前查看各欄位的資料型態
# =============================================================================
col_tpye = data_features.dtypes
# =============================================================================



# 查看資料集的描述性統計資料
# =============================================================================
columns = data_features.columns
stat_list = []

for col in columns:
    stat = data_features[col].describe(include="all")
    stat_list.append(stat)

data_summary = pd.concat(stat_list, axis=1)
data_summary = data_summary.reset_index()

# 移除暫存區資料
del col, columns, stat, stat_list

# save
data_summary = data_summary.to_csv('data/data_summary.csv', index = True, encoding = "big5")
# =============================================================================



# 再檢查一次缺失值
null2_data_features = data_features.isnull().sum()



# 儲存
# =============================================================================
data_features.to_csv('data/data_features.csv', index = True, encoding = "big5")
# =============================================================================


