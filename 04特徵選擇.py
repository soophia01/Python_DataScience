import pandas as pd
from sklearn.model_selection import train_test_split




# read train data
# =============================================================================
data_features2 = pd.read_csv("data/data_features2.csv",
                          encoding = "big5")
train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")

## 把 train data 從 data_features_2 抽出
train_feature2 = data_features2[0:46482]

## 移除 key 跟 index 欄位
train_feature2 = train_feature2.drop(columns=['Key', 'Index'])

train_label = train_label.drop(columns=['Index'])
# =============================================================================



# 特徵選擇
# =============================================================================
# RF
# ------------------------------------
# 建立並訓練 RF 模型
# clf = RandomForestClassifier()
# clf.fit(X_train_imputed, train_label)

# 查看特徵重要性
# importances = clf.feature_importances_

# 顯示每個特徵的重要性
# for feature, importance in zip(feature_names, importances):
#     print(f'{feature}: {importance:.2f}')
####  MemoryError: could not allocate 48739909632 bytes
# ------------------------------------


## SVM
# ------------------------------------
# 訓練 SVM 模型
# train_label = train_label["price_per_ping"]

# X_train, X_test, y_train, y_test = train_test_split(X_train, train_label, test_size=0.2, random_state=42)

# svm = SVC(kernel='linear')
# svm.fit(X_train, train_label)

# # 建立 SelectFromModel 物件，並將訓練好的 SVM 模型傳入。
# selector = SelectFromModel(svm)

# # 使用 fit_transform 方法將訓練資料傳入 selector，並傳回重要特徵的欄位索引。
# X_important_train = selector.fit_transform(X_train, y_train)


# # 嘗試在測試資料上使用重要特徵來評估模型的表現。
# X_important_test = selector.transform(X_test)

# score = svm.score(X_important_test, y_test)
# print(score)
# ------------------------------------
# =============================================================================




