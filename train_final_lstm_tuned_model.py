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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = 'features_continuous_unfiltered.csv'
TIME_STEPS = 60
EPOCHS = 20
BATCH_SIZE = 32

def build_lstm_transformer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
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
    print("Bắt đầu quy trình huấn luyện cuối cùng trên TOÀN BỘ DỮ LIỆU...")
    OUTPUT_DIR = 'final_lstm_tuned_model_artifacts'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    EPOCHS = 20
    BATCH_SIZE = 32
    df_raw = pd.read_csv(DATA_FILE)
    df_filtered = df_raw[~df_raw['Action Label'].isin(['None', '0', 'nan', np.nan])].copy()
    label_encoder = LabelEncoder()
    df_filtered['Action Label Encoded'] = label_encoder.fit_transform(df_filtered['Action Label'])
    num_classes = len(label_encoder.classes_)
    print("Các lớp hành vi sẽ được huấn luyện:", list(label_encoder.classes_))
    feature_cols = [col for col in df_raw.columns if col not in ['subject_id', 'Action Label', 'timestamp', 'Action Label Encoded']]
    scaler = StandardScaler()
    df_filtered[feature_cols] = pd.DataFrame(
    scaler.fit_transform(df_filtered[feature_cols]),
    columns=feature_cols,
    index=df_filtered.index
)
    X_list, y_list = [], []
    for sid in df_filtered['subject_id'].unique():
        subj_df = df_filtered[df_filtered['subject_id'] == sid]
        Xi, yi = create_sequences(subj_df[feature_cols].values, subj_df['Action Label Encoded'].values)
        if len(Xi) > 0:
            X_list.append(Xi)
            y_list.append(yi)
    X_data = np.concatenate(X_list)
    y_data = np.concatenate(y_list)
    class_weight_dict = dict(zip(np.unique(y_data), compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)))
    input_shape = (X_data.shape[1], X_data.shape[2])
    model = build_lstm_transformer_model(input_shape, num_classes)
    model.summary()
    print(f"\nHuấn luyện mô hình trên {len(X_data)} chuỗi từ tất cả các đối tượng...")
    model.fit(X_data, y_data, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight_dict, verbose=1)
    print(f"\nLưu model và các công cụ vào thư mục '{OUTPUT_DIR}'...")
    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, 'feature_columns.joblib'))
    print("--- HOÀN TẤT! ---")
    print(f"Mô hình cuối cùng đã sẵn sàng trong thư mục '{OUTPUT_DIR}'.")
