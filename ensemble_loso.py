import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

# --- Cấu hình ---
TIME_STEPS = 60
DATA_FILE = 'features_continuous_unfiltered.csv'
HYBRID_DIR = 'saved_models/hybrid_tuned'
LSTM_DIR = 'saved_models/lstm_tuned'
FOLDS = [1, 2, 3, 5]  # Các subject_id tương ứng với fold
weight_hybrid = 0.48
weight_lstm = 0.52
assert abs(weight_hybrid + weight_lstm - 1.0) < 1e-6

# --- Hàm tạo chuỗi ---
def create_sequences(X, y=None, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps])
        if y is not None:
            ys.append(y[i + time_steps - 1])
    return np.array(Xs), (np.array(ys) if y is not None else None)

# --- Load model & scaler theo fold ---
def load_fold_model_and_scaler(model_dir, fold_id):
    model_path = os.path.join(model_dir, f"best_model_fold_{fold_id}.keras")
    scaler_path = os.path.join(model_dir, f"scaler_fold_{fold_id}.joblib")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --- Load feature_cols và encoder (chung) ---
def load_common_config(model_dir):
    with open(os.path.join(model_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    return feature_cols, encoder

# --- Đọc dữ liệu và chuẩn hóa nhãn ---
df_raw = pd.read_csv(DATA_FILE)
df_raw = df_raw.sort_values('subject_id').reset_index(drop=True)
df_filtered = df_raw[~df_raw['Action Label'].isin(['None', '0', 'nan', np.nan])].copy()

global_label_encoder = LabelEncoder()
df_filtered['Action Label Encoded'] = global_label_encoder.fit_transform(df_filtered['Action Label'])

results = defaultdict(list)
print(f"\n🏁 Bắt đầu LOSO Ensemble với Weighted Soft Voting (Hybrid={weight_hybrid}, LSTM={weight_lstm})")

# --- Vòng lặp qua từng fold ---
for fold_id in FOLDS:
    print(f"\n=== Fold {fold_id} | Test subject: {fold_id} ===")
    
    df_test = df_filtered[df_filtered['subject_id'] == fold_id]
    df_train = df_filtered[df_filtered['subject_id'] != fold_id]

    # Load model và scaler riêng cho từng fold
    model_hybrid, scaler_hybrid = load_fold_model_and_scaler(HYBRID_DIR, fold_id)
    model_lstm, scaler_lstm = load_fold_model_and_scaler(LSTM_DIR, fold_id)

    # Load feature_cols và encoder dùng chung
    feature_cols, encoder = load_common_config(HYBRID_DIR)

    # Chuẩn hoá và tạo chuỗi
    X_hybrid = scaler_hybrid.transform(df_test[feature_cols])
    X_seq_hybrid, y_true = create_sequences(X_hybrid, df_test['Action Label Encoded'].values, TIME_STEPS)

    X_lstm = scaler_lstm.transform(df_test[feature_cols])
    X_seq_lstm, _ = create_sequences(X_lstm, None, TIME_STEPS)

    # Dự đoán xác suất
    proba_hybrid = model_hybrid.predict(X_seq_hybrid, verbose=0)
    proba_lstm = model_lstm.predict(X_seq_lstm, verbose=0)

    # Weighted average (Soft voting)
    avg_proba = weight_hybrid * proba_hybrid + weight_lstm * proba_lstm
    y_pred = np.argmax(avg_proba, axis=1)

    # Đánh giá
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"✅ Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, digits=3))

    results['accuracy'].append(acc)
    results['f1'].append(f1)

# --- Tổng kết ---
print("\n📊 TỔNG KẾT ENSEMBLE LOSO:")
print(f"Mean Accuracy: {np.mean(results['accuracy']):.4f}")
print(f"Mean F1-score:  {np.mean(results['f1']):.4f}")
