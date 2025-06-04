import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import glob
import os

# 讀取 train_info
train_info = pd.read_csv('39_Training_Dataset/train_info.csv')

# 特徵工程函數：統計特徵
def extract_features(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(-1, 6)  # 確保是 2D
    except Exception as e:
        print(f"❌ 讀取錯誤：{file_path}, 錯誤訊息: {e}")
        return None

    features = {}
    columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i, col in enumerate(columns):
        series = data[:, i]
        features[f'{col}_mean'] = np.mean(series)
        features[f'{col}_std'] = np.std(series)
        features[f'{col}_max'] = np.max(series)
        features[f'{col}_min'] = np.min(series)
    return features

# 批次處理訓練檔案
feature_list = []
train_files = glob.glob('39_Training_Dataset/train_data/*.txt')
for file in train_files:
    uid = os.path.basename(file).replace('.txt', '')
    feats = extract_features(file)
    feats['unique_id'] = int(uid)
    feature_list.append(feats)

feature_df = pd.DataFrame(feature_list)
print(feature_df.dtypes)
print(feature_df.head())
# 合併特徵與標籤
data = pd.merge(feature_df, train_info, on='unique_id')

# 目標欄位設定
targets = {
    'gender': 'gender',
    'hold racket handed': 'hold racket handed',
    'years': 'play years',
    'level': 'level'
}

# 取得特徵欄位
feature_cols = data.select_dtypes(include=['float64', 'float32']).columns.tolist()

# 訓練模型
models = {}
for target_name, target_col in targets.items():
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass' if target_name in ['years', 'level'] else 'Logloss',
        verbose=0,
        random_seed=42,
        task_type = 'GPU',
        devices = '0'
    )
    model.fit(data[feature_cols], data[target_col])
    models[target_name] = model

# 處理測試集
test_files = sorted(glob.glob('39_Test_Dataset/test_data/*.txt'))
test_features = []
for file in test_files:
    uid = os.path.basename(file).replace('.txt', '')
    feats = extract_features(file)
    feats['unique_id'] = int(uid)
    test_features.append(feats)

test_df = pd.DataFrame(test_features)
X_test = test_df.drop(['unique_id'], axis=1)

# 預測
submission = pd.DataFrame()
submission['unique_id'] = test_df['unique_id']

# 二元分類處理
for target in ['gender', 'hold racket handed']:
    cls_idx = models[target].classes_.tolist().index(1)
    submission[target] = models[target].predict_proba(X_test)[:, cls_idx]

# 球齡（三分類）
years_proba = models['years'].predict_proba(X_test)
for i, cls in enumerate(models['years'].classes_):
    submission[f'play years_{cls}'] = years_proba[:, i]

# 等級（四分類）
level_proba = models['level'].predict_proba(X_test)
for i, cls in enumerate(models['level'].classes_):
    submission[f'level_{cls}'] = level_proba[:, i]

submission = submission.round(5)

# 輸出結果
submission.to_csv('submission.csv', index=False)
print("✅ 預測完成，結果已儲存至 submission.csv")
