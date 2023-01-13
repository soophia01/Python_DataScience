from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 建立 RFR 模型
model = RandomForestRegressor()

# 設定要搜尋的參數
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# 建立網格搜尋物件
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)

# 開始搜尋最佳參數
grid_result = grid_search.fit(X_train, y_train)

# 印出最佳參數
print("Best Parameters: ", grid_result.best_params_)

# 印出最佳結果的平均交叉驗證分數
print("Best Cross-Validation Score: ", grid_result.best_score_)
