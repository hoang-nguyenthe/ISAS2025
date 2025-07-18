#Hybrid model
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib, json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dropout, Dense, Bidirectional,
    MultiHeadAttention, LayerNormalization, Add,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = 'features_continuous_unfiltered.csv'
TIME_STEPS = 60
EPOCHS = 20
BATCH_SIZE = 32

def build_lstm_transformer_model(input_shape, num_classes,
                                 lstm_units=128,
                                 num_heads=4,
                                 ff_dim=128,
                                 num_transformer_blocks=1,
                                 dropout_rate=0.3,
                                 dense_units=64,
                                 learning_rate=1e-4):
    """
    Xây dựng mô hình kết hợp Bi-LSTM làm lớp trích xuất đặc trưng tuần tự,
    và một khối Transformer để tìm các mối quan hệ toàn cục.
    """
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inp)
    x = Dropout(dropout_rate)(x)
    for _ in range(num_transformer_blocks):
        x1 = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=lstm_units * 2
        )(x1, x1)
        attention_output = Dropout(dropout_rate)(attention_output)
        x2 = Add()([x, attention_output])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        ffn = Dense(ff_dim, activation="relu")(x3)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(lstm_units * 2)(ffn)
        x = Add()([x2, ffn])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)
if __name__ == "__main__":
    print("Bắt đầu quy trình huấn luyện và đánh giá LOSO (Hybrid: Bi-LSTM + Transformer)...")
    df_raw = pd.read_csv(DATA_FILE)
    ARTIFACT_DIR= 'saved_models/hybrid_tuned'
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df_filtered = df_raw.dropna(subset=['Action Label'])
    valid_labels = df_filtered[~df_filtered['Action Label'].isin(['None', '0', 'nan'])]['Action Label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    num_classes = len(label_encoder.classes_)
    print("Các lớp hành vi sẽ được huấn luyện:", list(label_encoder.classes_))
    joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, 'label_encoder.joblib'))
    feature_cols = [col for col in df_raw.columns if col not in ['subject_id', 'Action Label']]
    with open(os.path.join(ARTIFACT_DIR, 'feature_cols.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    subject_ids = df_raw['subject_id'].unique()
    results = []
    total_cm = np.zeros((num_classes, num_classes), dtype=int)
    for test_subject_id in tqdm(subject_ids, desc="Tổng quan các vòng lặp LOSO"):
        print(f"\n===== Vòng lặp: Để đối tượng {int(test_subject_id)} ra làm kiểm tra =====")
        train_df = df_raw[df_raw['subject_id'] != test_subject_id]
        test_df = df_raw[df_raw['subject_id'] == test_subject_id]
        feature_cols = [col for col in df_raw.columns if col not in ['subject_id', 'Action Label']]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])
        fold_scaler_path = os.path.join(ARTIFACT_DIR, f'scaler_fold_{int(test_subject_id)}.joblib')
        joblib.dump(scaler, fold_scaler_path)
        y_train_raw = train_df['Action Label'].values
        y_test_raw = test_df['Action Label'].values
        X_train_list, y_train_list = [], []
        for sid in train_df['subject_id'].unique():
            mask = np.where(train_df['subject_id'] == sid)[0]
            Xi, yi = create_sequences(X_train_scaled[mask], y_train_raw[mask])
            if len(Xi) > 0:
                X_train_list.append(Xi)
                y_train_list.append(yi)
        X_train_seq = np.concatenate(X_train_list)
        y_train_seq = np.concatenate(y_train_list)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw)
        train_mask = pd.Series(y_train_seq).notna() & (~pd.Series(y_train_seq).isin(['None', '0', 'nan']))
        X_train_seq_filtered = X_train_seq[train_mask]
        y_train_seq_filtered = y_train_seq[train_mask]
        test_mask = pd.Series(y_test_seq).notna() & (~pd.Series(y_test_seq).isin(['None', '0', 'nan']))
        X_test_seq_filtered = X_test_seq[test_mask]
        y_test_seq_filtered = y_test_seq[test_mask]
        y_train_encoded = label_encoder.transform(y_train_seq_filtered)
        y_test_encoded = label_encoder.transform(y_test_seq_filtered)
        classes = np.unique(y_train_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_encoded)
        class_weight_dict = dict(zip(classes, weights))
        print("Class weights:", class_weight_dict)
        input_shape = (X_train_seq_filtered.shape[1], X_train_seq_filtered.shape[2])
        model = build_lstm_transformer_model(input_shape, num_classes)
        print(f"Mô hình sẽ dự đoán trên {model.output_shape[1]} lớp.")
        print(f"Huấn luyện mô hình trên {len(X_train_seq_filtered)} chuỗi sạch...")
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        checkpoint_filepath = f'saved_models/hybrid_tuned/best_model_fold_{int(test_subject_id)}.keras'
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        model.fit(X_train_seq_filtered, y_train_encoded, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, class_weight=class_weight_dict, callbacks=[early_stop,model_checkpoint], verbose=1)
        print(f"Tải lại model tốt nhất từ: {checkpoint_filepath}")
        best_model = tf.keras.models.load_model(checkpoint_filepath)
        y_pred = np.argmax(best_model.predict(X_test_seq_filtered), axis=1)
        print("Đánh giá mô hình trên dữ liệu kiểm tra...")
        if len(X_test_seq_filtered) > 0:
            y_pred = np.argmax(best_model.predict(X_test_seq_filtered), axis=1)
            cm = confusion_matrix(y_test_encoded, y_pred, labels=np.arange(num_classes))
            total_cm += cm
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
            print(f"Kết quả cho vòng kiểm tra {int(test_subject_id)}:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Macro F1-Score: {f1:.4f}")
            print("\nBáo cáo chi tiết:")
            print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0))
            results.append({'fold': test_subject_id, 'accuracy': accuracy, 'f1_score': f1})
        else:
            print(f"Không có dữ liệu kiểm tra hợp lệ cho fold {int(test_subject_id)}.")
    print("\n===== KẾT QUẢ ĐÁNH GIÁ LOSO TRUNG BÌNH (ĐÁNG TIN CẬY NHẤT) =====")
    if results:
        avg_accuracy = np.mean([res['accuracy'] for res in results])
        avg_f1 = np.mean([res['f1_score'] for res in results])
        print(f"Accuracy trung bình: {avg_accuracy:.4f}")
        print(f"Macro F1-Score trung bình: {avg_f1:.4f}")
        plt.figure(figsize=(12, 10))
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Tổng hợp Ma trận nhầm lẫn từ tất cả các vòng LOSO')
        plt.ylabel('Nhãn thực tế (True Label)')
        plt.xlabel('Nhãn dự đoán (Predicted Label)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("Không có kết quả nào để tổng hợp.")

