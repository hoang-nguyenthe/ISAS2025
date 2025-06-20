import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler

#load data 
df"labels = pd.read_csv(".csv") //update file csv file path

# ==== STEP 2: CREATE SLIDING WINDOWS ====
def create_windows(df, window_size=30, stride=15):
    X, y, subject_ids = [], [], []
    for sid in df['subject_id'].unique():
        df_subj = df[df['subject_id'] == sid].reset_index(drop=True)
        for i in range(0, len(df_subj) - window_size + 1, stride):
            window = df_subj.iloc[i:i+window_size]
            features = window.drop(columns=['frame_id', 'subject_id', 'label']).values.flatten()
            label = window['label'].mode()[0]
            X.append(features)
            y.append(label)
            subject_ids.append(sid)
    return np.array(X), np.array(y), np.array(subject_ids)

X, y, groups = create_windows(df)

# ==== STEP 3: ENCODE LABELS ====
le = LabelEncoder()
y_enc = le.fit_transform(y)

#Evaluate using LOSO
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y_enc, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    model = SVC(kernel='rbf', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))