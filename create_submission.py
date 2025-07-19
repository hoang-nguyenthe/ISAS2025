import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

FEATURE_FILE = "features_test.csv"
FRAME_FILE = "test data_keypoint.csv"
OUTPUT_FILE = "Binary_Phoenix_test.csv"
LSTM_DIR = "final_lstm_tuned_model_artifacts"
HYBRID_DIR = "final_hybrid_tuned_model_artifacts"
WEIGHT_LSTM = 0.6
WEIGHT_HYBRID = 0.4
TIME_STEPS = 60

def create_sequences(X, time_steps=60):
    Xs = []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

def load_model_artifacts(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "final_model.keras"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    encoder = joblib.load(os.path.join(model_dir, "encoder.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return model, scaler, encoder, feature_cols

def main():
    print("🚀 Ensemble Final Submission – START")
    df_feat = pd.read_csv(FEATURE_FILE)
    df_raw = pd.read_csv(FRAME_FILE)
    assert len(df_feat) == len(df_raw), "❌ Không khớp số dòng giữa features và frame_id!"
    model_lstm, scaler_lstm, encoder_lstm, feat_cols_lstm = load_model_artifacts(LSTM_DIR)
    model_hybrid, scaler_hybrid, encoder_hybrid, feat_cols_hybrid = load_model_artifacts(HYBRID_DIR)
    X_input_lstm = df_feat[feat_cols_lstm].copy()
    X_input_hybrid = df_feat[feat_cols_hybrid].copy()
    X_scaled_lstm = scaler_lstm.transform(X_input_lstm)
    X_scaled_hybrid = scaler_hybrid.transform(X_input_hybrid)
    X_seq_lstm = create_sequences(X_scaled_lstm, TIME_STEPS)
    X_seq_hybrid = create_sequences(X_scaled_hybrid, TIME_STEPS)
    probs_lstm = model_lstm.predict(X_seq_lstm, verbose=0)
    probs_hybrid = model_hybrid.predict(X_seq_hybrid, verbose=0)
    probs_final = WEIGHT_LSTM * probs_lstm + WEIGHT_HYBRID * probs_hybrid
    preds_encoded = np.argmax(probs_final, axis=1)
    pred_labels = encoder_hybrid.inverse_transform(preds_encoded)
    valid_indices = df_feat.index[TIME_STEPS - 1:]
    df_preds = pd.DataFrame({
        "timestamp": df_raw.loc[valid_indices, "frame_id"].values,
        "predicted_label": pred_labels
    }, index=valid_indices)
    df_out = df_raw[["frame_id"]].copy()
    df_out.rename(columns={"frame_id": "timestamp"}, inplace=True)
    df_out["predicted_label"] = df_preds["predicted_label"]
    df_out["predicted_label"] = df_out["predicted_label"].bfill().ffill().fillna("None")
    df_out["participant_id"] = 4
    df_out[["participant_id", "timestamp", "predicted_label"]].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ File submission đã lưu: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
