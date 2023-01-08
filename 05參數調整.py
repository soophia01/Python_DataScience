import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV # 自動對一組參數進行評估，並找到最佳的參數組合



# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")
train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

## 把 train data 從 data_features_2 抽出
data_features3 = data_features3[0:46482]

## 移除 key 跟 index 欄位
data_features3 = data_features3.drop(columns=['Key', 'Index'])

train_label = train_label.drop(columns=['Index'])
# =============================================================================



# 使用驗證集來調整 SVM 參數
# =============================================================================
# 定義 SVM 模型
model = SVC()

# 定义参数网格
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}

# 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(data_features3, train_label)

# 输出最优参数
print(grid_search.best_params_)
# =============================================================================




