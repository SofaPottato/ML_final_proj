import pandas as pd
import numpy as np
import glob, os
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 特徵提取（略，與前面相同）
def extract_features(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 6)
    feats = {}
    cols = ['Ax','Ay','Az','Gx','Gy','Gz']
    for i,c in enumerate(cols):
        s = data[:,i]
        feats[f'{c}_mean'] = s.mean()
        feats[f'{c}_std']  = s.std()
        feats[f'{c}_max']  = s.max()
        feats[f'{c}_min']  = s.min()
    return feats

# 2. 讀訓練特徵 & 標籤
# ─────────────────
# 批次讀 train_data/*.txt
train_feats = []
for f in glob.glob('39_Training_Dataset/train_data/*.txt'):
    uid = int(os.path.basename(f).replace('.txt',''))
    feats = extract_features(f)
    feats['unique_id'] = uid
    train_feats.append(feats)
train_feat_df = pd.DataFrame(train_feats)

# 讀 train_info.csv，合併
train_info = pd.read_csv('39_Training_Dataset/train_info.csv')
train_df   = pd.merge(train_feat_df, train_info, on='unique_id')

# 3. 讀測試特徵
# ─────────────────
test_feats = []
test_files = sorted(glob.glob('39_Test_Dataset/test_data/*.txt'))
for f in test_files:
    uid = int(os.path.basename(f).replace('.txt',''))
    feats = extract_features(f)
    feats['unique_id'] = uid
    test_feats.append(feats)
test_df = pd.DataFrame(test_feats)

# 4. 依序對每個 target 做 Optuna + 最終訓練 + 預測
# ─────────────────────────────────────────────
targets = {
    'gender':               'gender',
    'hold_racket_handed':   'hold racket handed',
    'play_years':           'play years',
    'level':                'level'
}

feature_cols = train_df.select_dtypes(include=['float64','float32']).columns.tolist()
submission   = pd.DataFrame({'unique_id': test_df['unique_id']})

def compute_metric(y_true, proba, classes):
    """二元：return roc_auc_score(y, proba[:,1])
       多元：return roc_auc_score(y, proba, multi_class='ovr', average='micro')"""
    if len(classes) == 2:
        return roc_auc_score(y_true, proba[:,1])
    else:
        return roc_auc_score(y_true, proba, multi_class='ovr', average='micro')

for name, col in targets.items():
    print(f"\n▶▶▶ Optimizing & training for [{col}] …")
    y = train_df[col]
    X = train_df[feature_cols]

    # Optuna objective
    def objective(trial):
        params = {
            'iterations':    trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'depth':         trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1e-2, 100, log=True),
            'loss_function': 'MultiClass' if len(y.unique())>2 else 'Logloss',
            'task_type':     'GPU',
            'devices':       '0',
            'verbose':       False,
            'random_seed':   42
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                use_best_model=False,
                logging_level='Silent'
            )
            proba = model.predict_proba(X_va)
            scores.append(compute_metric(y_va, proba, model.classes_))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', study_name=f'study_{name}')
    study.optimize(objective, n_trials=30, timeout=300)

    print(f"→ Best CV metric for [{col}]: {study.best_value:.4f}")
    print("→ Best params:", study.best_params)

    # 用最佳參數訓練全資料
    bestp = study.best_params.copy()
    bestp.update({
        'loss_function': 'MultiClass' if len(y.unique())>2 else 'Logloss',
        'task_type':     'GPU',
        'devices':       '0',
        'verbose':       False,
        'random_seed':   42
    })
    final_model = CatBoostClassifier(**bestp)
    final_model.fit(X, y)

    # 預測測試集
    proba_test = final_model.predict_proba(test_df[feature_cols])
    classes    = final_model.classes_
    if len(classes) == 2:
        submission[name] = proba_test[:,1]
    else:
        for i, cls in enumerate(classes):
            submission[f'{col}_{cls}'] = proba_test[:, i]

# 5. 輸出 submission.csv
# ─────────────────
submission = submission.sort_values('unique_id').reset_index(drop=True)
submission.to_csv('submission.csv', index=False, float_format='%.5f')
print("\n✅ 已生成 submission.csv，包含所有目標的預測機率／多分類機率欄位。")
