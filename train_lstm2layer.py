
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional, Multiply, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
ARTIFACT_DIR = 'saved_models/lstm'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATA_FILE = 'features_continuous_unfiltered.csv'
TIME_STEPS = 60
EPOCHS = 20
BATCH_SIZE = 32
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.tanh(tf.matmul(inputs, self.W))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return tf.reduce_sum(context_vector, axis=1)
def build_lstm_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)]); ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

if __name__ == "__main__":
    print("Bắt đầu quy trình huấn luyện và đánh giá LOSO...")
    df_raw = pd.read_csv(DATA_FILE)
    df_filtered = df_raw.dropna(subset=['Action Label'])
    valid_labels = df_filtered[~df_filtered['Action Label'].isin(['None', '0', 'nan'])]['Action Label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    num_classes = len(label_encoder.classes_)
    print("Các lớp hành vi sẽ được huấn luyện:", list(label_encoder.classes_))
    joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, 'encoder.joblib'))
    feature_cols = [col for col in df_raw.columns if col not in ['subject_id', 'Action Label']]
    joblib.dump(feature_cols, os.path.join(ARTIFACT_DIR, 'feature_columns.joblib'))
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
        y_train = train_df['Action Label'].values
        y_test = test_df['Action Label'].values
        X_train_list, y_train_list = [], []
        for sid in train_df['subject_id'].unique():
            mask = train_df['subject_id'] == sid
            Xi, yi = create_sequences(
                X_train_scaled[mask.values],
                y_train[mask.values],
                TIME_STEPS
            )
            X_train_list.append(Xi)
            y_train_list.append(yi)

        if X_train_list:
            X_train_seq = np.vstack(X_train_list)
            y_train_seq = np.concatenate(y_train_list)
        else:
            X_train_seq = np.empty((0, TIME_STEPS, len(feature_cols)))
            y_train_seq = np.empty((0,))

        X_test_list, y_test_list = [], []
        for sid in test_df['subject_id'].unique():
            mask = test_df['subject_id'] == sid
            Xi, yi = create_sequences(
                X_test_scaled[mask.values],
                y_test[mask.values],
                TIME_STEPS
            )
            X_test_list.append(Xi)
            y_test_list.append(yi)
        if X_test_list:
            X_test_seq = np.vstack(X_test_list)
            y_test_seq = np.concatenate(y_test_list)
        else:
            X_test_seq = np.empty((0, TIME_STEPS, len(feature_cols)))
            y_test_seq = np.empty((0,))
        y_train_series = pd.Series(y_train_seq)
        train_mask = (~y_train_series.isin(['None', '0', 'nan'])) & (y_train_series.notna())
        X_train_seq_filtered = X_train_seq[train_mask.values]
        y_train_seq_filtered = y_train_seq[train_mask.values]
        y_test_series = pd.Series(y_test_seq)
        test_mask = (~y_test_series.isin(['None', '0', 'nan'])) & (y_test_series.notna())
        X_test_seq_filtered = X_test_seq[test_mask.values]
        y_test_seq_filtered = y_test_seq[test_mask.values]
        y_train_encoded = label_encoder.transform(y_train_seq_filtered)
        y_test_encoded = label_encoder.transform(y_test_seq_filtered)
        classes = np.unique(y_train_encoded)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_encoded
        )
        class_weight_dict = dict(zip(classes, weights))
        print("Class weights:", class_weight_dict)
        input_shape = (X_train_seq_filtered.shape[1], X_train_seq_filtered.shape[2])
        model = build_lstm_model(input_shape, num_classes)
        print(f"Mô hình sẽ dự đoán trên {model.output_shape[1]} lớp.")
        print(f"Huấn luyện mô hình trên {len(X_train_seq_filtered)} chuỗi sạch...")
        checkpoint_filepath = os.path.join(ARTIFACT_DIR, f'best_model_fold_{int(test_subject_id)}.keras')
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_ckpt = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True
)
        model.fit(
            X_train_seq_filtered, y_train_encoded,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=[early_stop, model_ckpt],
            verbose=1
)
        print("Đánh giá mô hình trên dữ liệu kiểm tra...")
        best_model = tf.keras.models.load_model(
            checkpoint_filepath)
        proba = best_model.predict(X_test_seq_filtered, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        cm = confusion_matrix(y_test_encoded, y_pred, labels=np.arange(num_classes))
        total_cm += cm
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='macro')
        print(f"Kết quả cho vòng kiểm tra {int(test_subject_id)}:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Macro F1-Score: {f1:.4f}")
        print("\nBáo cáo chi tiết:")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0))
        results.append({'fold': test_subject_id, 'accuracy': accuracy, 'f1_score': f1})
    print("\n===== KẾT QUẢ ĐÁNH GIÁ LOSO TRUNG BÌNH MÔ HÌNH 2 LỚP LSTM =====")
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
