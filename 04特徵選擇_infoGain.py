import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("data/data_features4.csv",
                          encoding = "big5")
train_label = pd.read_csv("data/train_label.csv",
                          encoding = "big5")
# 移除 key 跟 index 欄位
df = df.drop(columns=['Key', 'Index'])

X = df[0:46482]
y = train_label["price_per_ping"]

# Create an instance of SelectKBest
selector = SelectKBest(mutual_info_classif, k=10)

# Fit the selector to your data
selector.fit(X, y)

# Get the selected features
selected_features = selector.transform(X)

# check
# Get a boolean mask of the selected features
selected_mask = selector.get_support()

# Print the selected features' names
feature_names = X.columns
selected_features = feature_names[selected_mask]
print(selected_features)

# ['主要建材', '主要用途', '交易標的', '土地_log_2', '土地', '建物_log_2', '建物', '建物現況格局-廳', '建物現況格局-隔間', '都市土地使用分區']