import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from collections import defaultdict

TIME_STEPS = 60
DATA_FILE = 'features_continuous_unfiltered.csv'
HYBRID_DIR = 'saved_models/hybrid_tuned'
LSTM_DIR = 'saved_models/lstm_tuned'
FOLDS = [1, 2, 3, 5]

def create_sequences(X, y=None, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps])
        if y is not None:
            ys.append(y[i + time_steps - 1])
    return np.array(Xs), (np.array(ys) if y is not None else None)

def load_fold_model_and_scaler(model_dir, fold_id):
    model_path = os.path.join(model_dir, f"best_model_fold_{fold_id}.keras")
    scaler_path = os.path.join(model_dir, f"scaler_fold_{fold_id}.joblib")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_common_config(model_dir):
    with open(os.path.join(model_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    abnormal_labels = ['Attacking', 'Throwing things', 'Head banging', 'Biting nails']
    abnormal_class_indices = [encoder.transform([lbl])[0] for lbl in abnormal_labels]
    return feature_cols, encoder, abnormal_class_indices

def preload_fold_predictions(df_filtered, feature_cols, encoder):
    all_fold_results = {}
    for fold_id in FOLDS:
        df_test = df_filtered[df_filtered['subject_id'] == fold_id]
        labels = encoder.transform(df_test['Action Label'])
        model_hybrid, scaler_hybrid = load_fold_model_and_scaler(HYBRID_DIR, fold_id)
        model_lstm, scaler_lstm = load_fold_model_and_scaler(LSTM_DIR, fold_id)
        X_hybrid = scaler_hybrid.transform(df_test[feature_cols])
        X_seq_hybrid, y_true = create_sequences(X_hybrid, labels, TIME_STEPS)
        X_lstm = scaler_lstm.transform(df_test[feature_cols])
        X_seq_lstm, _ = create_sequences(X_lstm, None, TIME_STEPS)
        proba_hybrid = model_hybrid.predict(X_seq_hybrid, verbose=0)
        proba_lstm = model_lstm.predict(X_seq_lstm, verbose=0)
        all_fold_results[fold_id] = {
            'y_true': y_true,
            'proba_hybrid': proba_hybrid,
            'proba_lstm': proba_lstm
        }
    return all_fold_results

def evaluate_weight(w_hybrid, all_fold_results):
    w_lstm = round(1.0 - w_hybrid, 2)
    f1_scores = []
    for fold_id in FOLDS:
        data = all_fold_results[fold_id]
        avg_proba = w_hybrid * data['proba_hybrid'] + w_lstm * data['proba_lstm']
        y_pred = np.argmax(avg_proba, axis=1)
        weights = np.ones_like(data['y_true'], dtype=np.float32)
        for ab_cls in abnormal_class_indices:
            weights[data['y_true'] == ab_cls] *= 3.0
        f1 = f1_score(data['y_true'], y_pred, average='weighted', sample_weight=weights)
        f1_scores.append(f1)
    return w_hybrid, w_lstm, np.mean(f1_scores)

if __name__ == "__main__":
    print("🚀 Bắt đầu Grid Search (KHÔNG dùng multiprocessing)...")
    df_raw = pd.read_csv(DATA_FILE)
    df_raw = df_raw.sort_values('subject_id').reset_index(drop=True)
    df_filtered = df_raw[~df_raw['Action Label'].isin(['None', '0', 'nan', np.nan])].copy()
    feature_cols, encoder, abnormal_class_indices = load_common_config(HYBRID_DIR)
    df_filtered['Action Label Encoded'] = encoder.transform(df_filtered['Action Label'])
    all_fold_results = preload_fold_predictions(df_filtered, feature_cols, encoder)
    weight_list = [round(w, 2) for w in np.arange(0.4, 0.6, 0.02)]
    results = []
    for w in weight_list:
        w_hybrid, w_lstm, mean_f1 = evaluate_weight(w, all_fold_results)
        results.append((w_hybrid, w_lstm, mean_f1))
        print(f"Hybrid={w_hybrid:.2f}, LSTM={w_lstm:.2f} → F1 = {mean_f1:.4f}")
    best = max(results, key=lambda x: x[2])
    print("\n🏆 Trọng số tốt nhất:")
    print(f"Hybrid = {best[0]:.2f}, LSTM = {best[1]:.2f} → F1 = {best[2]:.4f}")
