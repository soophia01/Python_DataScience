import pandas as pd
import numpy as np
from sklearn.svm import SVC


# read train data
# =============================================================================
data_features3 = pd.read_csv("data/data_features3.csv",
                          encoding = "big5")
train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")


# train
## 把 train data 從 data_features2 抽出
X_train = data_features3[0:46482]
## 移除 key 跟 index 欄位
X_train = X_train.drop(columns=['Key', 'Index'])

y_train = train_label.drop(columns=['Index'])
## 轉一維
y_train_array = y_train.values
y_train_array = np.ravel(y_train_array)

# test
## 把 test data 從 data_features2 抽出
X_test = data_features3[46482:]

## 移除 key 跟 index 欄位
X_test = X_test.drop(columns=['Key', 'Index'])
# =============================================================================



# SVM
# =============================================================================
svm = SVC()
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
# =============================================================================



# =============================================================================
# 
# 
# # https://ithelp.ithome.com.tw/articles/10272586
# 
# # Step 1: Import libraries
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# 
# # Step 2: Load and preprocess the data
# X = np.load('train_feature.npy')
# y = np.load('train_label.npy')
# 
# # Step 3: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Step 4: Create and train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# 
# # Step 5: Evaluate the model
# # 不要用這個，自己手動切助教給的training set
# accuracy = model.score(X_test, y_test)
# print('Accuracy:', accuracy)
# 
# # Step 6: Fine-tune the model (optional)
# parameters = {'max_depth': [2, 4, 6, 8], 'n_estimators': [10, 50, 100, 200]}
# clf = GridSearchCV(model, parameters)
# clf.fit(X_train, y_train)
# 
# =============================================================================
