import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel




# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")
train_label2 = pd.read_csv("data/train_label2.csv",
                          encoding = "big5")

# train
# -----------------------------
X_train = data_features3[0:46482]
# 移除 key 跟 index 欄位
X_train = X_train.drop(columns=['Key', 'Index'])
y_train = train_label2.drop(columns=['Index'])

# 把 y 轉一維
y_train_array = y_train.values
y_train_array = np.ravel(y_train_array)
# =============================================================================



# 特徵選擇
# =============================================================================
# SVM
# ------------------------------------
# 訓練 SVM 模型
train_label2 = train_label2["price_per_ping_log"]

X_train, X_test, y_train, y_test = train_test_split(X_train, train_label2, test_size = 0.2, random_state = 42)

svm = SVC(kernel='linear')
svm.fit(X_train, train_label2)

# 建立 SelectFromModel 物件，並將訓練好的 SVM 模型傳入。
selector = SelectFromModel(svm)

# 使用 fit_transform 方法將訓練資料傳入 selector，並傳回重要特徵的欄位索引。
X_important_train = selector.fit_transform(X_train, y_train)


# 嘗試在測試資料上使用重要特徵來評估模型的表現。
X_important_test = selector.transform(X_test)

score = svm.score(X_important_test, y_test)
print(score)
# ------------------------------------


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
# =============================================================================




