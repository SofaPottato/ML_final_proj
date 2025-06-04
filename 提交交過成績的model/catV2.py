import pandas as pd
import numpy as np
import glob, os
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

# 1) 讀取 train_info
train_info = pd.read_csv('39_Training_Dataset/train_info.csv')

# 2) 特徵工程函數
def extract_features(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 6)
    cols = ['Ax','Ay','Az','Gx','Gy','Gz']
    feats = {}
    for i, c in enumerate(cols):
        s = data[:, i]
        feats[f'{c}_mean'] = s.mean()
        feats[f'{c}_std']  = s.std()
        feats[f'{c}_max']  = s.max()
        feats[f'{c}_min']  = s.min()
    return feats

# 3) 批次抽特徵
feature_list = []
for fp in glob.glob('39_Training_Dataset/train_data/*.txt'):
    uid = int(os.path.basename(fp).replace('.txt',''))
    f = extract_features(fp)
    f['unique_id'] = uid
    feature_list.append(f)
feature_df = pd.DataFrame(feature_list)

# 4) 合併標籤
data = pd.merge(feature_df, train_info, on='unique_id')
feature_cols = data.select_dtypes(include=['float64','float32']).columns.tolist()

# 5) 目標欄位設定
targets = {
    'gender': 'gender',
    'hold racket handed': 'hold racket handed',
    'years': 'play years',
    'level': 'level'
}

# 6) Grid Search 參數表（可依需求增減）
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05],
    'iterations': [100, 200]
}

# 7) Cross-Validation 設定
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8) 準備 scorer
auc_binary = make_scorer(roc_auc_score, needs_proba=True)
def auc_multi(y_true, y_proba):
    return roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
auc_multi_scorer = make_scorer(auc_multi, needs_proba=True)

# 9) 每個 target 執行 GridSearchCV
models = {}
for name, col in targets.items():
    print(f"\n--- GridSearch for {name} ---")
    # loss / scorer 依 target 類型選擇
    loss = 'MultiClass' if name in ['years','level'] else 'Logloss'
    scorer = auc_multi_scorer if name in ['years','level'] else auc_binary

    base_clf = CatBoostClassifier(
        loss_function=loss,
        task_type='GPU',
        devices='0',
        random_seed=42,
        verbose=False
    )

    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring=scorer,
        cv=skf,
        n_jobs=1,         # GPU 模式下建議 1
        verbose=2,
        return_train_score=True
    )

    y = data[col]
    grid.fit(data[feature_cols], y)

    print(f"Best CV score for {name}: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    models[name] = grid.best_estimator_

# 10) 處理測試集
test_files = sorted(glob.glob('39_Test_Dataset/test_data/*.txt'))
t_feats = []
for fp in test_files:
    uid = int(os.path.basename(fp).replace('.txt',''))
    f = extract_features(fp)
    f['unique_id'] = uid
    t_feats.append(f)
test_df = pd.DataFrame(t_feats)
X_test  = test_df[feature_cols]

# 11) 預測並組 submission
submission = pd.DataFrame({'unique_id': test_df['unique_id']})

# 二元
for t in ['gender','hold racket handed']:
    probs = models[t].predict_proba(X_test)
    idx1  = models[t].classes_.tolist().index(1)
    submission[t] = probs[:, idx1]

# 多類：years
yp = models['years'].predict_proba(X_test)
for i, cls in enumerate(models['years'].classes_):
    submission[f'play years_{cls}'] = yp[:, i]

# 多類：level
zp = models['level'].predict_proba(X_test)
for i, cls in enumerate(models['level'].classes_):
    submission[f'level_{cls}'] = zp[:, i]

submission = submission.round(5)
submission.to_csv('submission.csv', index=False)
print("✅ 完成！最佳模型已套用，submission.csv 已輸出。")
