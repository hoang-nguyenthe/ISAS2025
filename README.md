# 🧠 ISAS 2025: Abnormal Behavior Recognition using Pose-Based Feature Engineering and Deep Ensemble Learning

## 📌 Project Summary

We propose an abnormal activity recognition system for individuals with developmental disabilities using 2D pose keypoint data and deep learning. The solution consists of:
- A **feature engineering pipeline** crafted from raw keypoints
- A **dual deep learning ensemble** combining Bi-LSTM and Transformer, optimized for detecting abrupt behaviors like *"Attacking"* or *"Throwing things"*

## 👥 Authors

The Hoang Nguyen, Gia Huy Ly, and Duy Khanh Dinh Hoang
All authors are currently studying at VNU-HCM, Ho Chi Minh City University of Technology (HCMUT).

## 🗂️ Dataset

The dataset is provided by the **ISAS 2025 challenge**:
- 4 subjects for training, 1 subject for testing using LOSO (Leave-One-Subject-Out)
- 8 labeled activities: 4 **normal** (e.g., *Sitting*, *Walking*) and 4 **unusual** (e.g., *Biting nails*, *Attacking*)
- Pose keypoints extracted via YOLOv7 at 30 FPS

Main challenges:
- **Data imbalance**: more normal than abnormal frames
- **Temporal variability** between activity types
- **Subject-specific differences** in motion styles
- **Short and unpredictable unusual behaviors** (e.g., *Attacking*)

## 🔧 Feature Engineering Pipeline

We designed over **70 continuous features per frame** from keypoints to capture motion, geometry, asymmetry, and temporal-frequency characteristics:

| Feature Group         | Description |
|-----------------------|-------------|
| Motion                | Velocity, acceleration, jerk of hands and nose |
| Geometric             | Euclidean distances (e.g., hand–nose), joint angles (elbow, knee), torso angle, hand-above-shoulder flag |
| Asymmetry             | Speed and position differences between left/right hands |
| Temporal statistics   | Rolling mean, std, and max (1.5s window ≈ 45 frames) |
| Frequency & regularity| Dominant frequency (FFT), zero-crossing rate (ZCR), movement regularity |

All features are computed on **interpolated and smoothed keypoints** to reduce noise.

## 🧠 Model Architecture

### 1. **Deep Bi-LSTM**
- Two Bi-LSTM layers (128, 64 units)
- BatchNorm and Dropout for generalization
- Effective for repetitive behaviors (*Walking*, *Sitting*)

### 2. **Hybrid: Bi-LSTM + Transformer**
- Bi-LSTM for short-term motion encoding
- Transformer for long-range, non-linear dependencies
- Effective for bursty behaviors (*Attacking*, *Throwing*)

### ⚖️ Ensemble Strategy
Softmax probability weighted average:
- Bi-LSTM: 52%
- Hybrid: 48%

Weights tuned via LOSO cross-validation.

## ⏱️ Temporal Settings

| Component                | Value                         |
|--------------------------|-------------------------------|
| Frame rate               | 30 FPS                        |
| Input sequence length    | 60 frames (≈ 2 seconds)       |
| Feature rolling window   | 45 frames (≈ 1.5 seconds)     |
| Overlap rate             | ~90%                          |
| Subjects used for training | 1, 2, 3, 5                  |

Sliding window segmentation ensures dense sampling for short-duration activities.

## 📊 Evaluation Strategy

### A. **Activity Classification**
- Input: unlabeled pose sequences
- Output: `participant_id, timestamp, predicted_label`
- Metrics: **Accuracy, Abnormal F1-Score, Precision, Recall**

### B. **LOSO Evaluation**
- Evaluate model generalization on unseen subject
- Submit LOSO-specific evaluation report

## 📊 LOSO Summary Results (Abnormal Behaviors)

Below is the average performance across all LOSO folds for abnormal behaviors using the Ensemble model (Bi-LSTM + Hybrid):

| Abnormal Behavior    | Average F1-score |
|----------------------|------------------|
| Attacking            | 70.25%           |
| Biting nails         | 56.65%           |
| Head banging         | 72.32%           |
| Throwing things      | 60.30%           |
## 📊 LOSO Summary Results (Normal Behaviors)

| Normal Behavior       | Average F1-score |
|-----------------------|------------------|
| Eating snacks         | 52.33%           |
| Sitting quietly       | 58.05%           |
| Using phone           | 11.6%            |
| Walking               | 93.65%           |


## 📁 Submission Files

- `[team_name]_test.csv`: format `[participant_id, timestamp, predicted_label]`


## 🚀 Key Contributions

- Engineered 70+ temporal and geometric features from 2D keypoints
- Designed a hybrid deep model suitable for both smooth and irregular behaviors
- Applied ensemble fusion for improved recognition accuracy
- Tuned temporal parameters using rolling window and LLM-guided prompting
- Achieved high performance on short, bursty and challenging behaviors

## 🛠️ Setup & Execution Guide

### 1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Step-by-step Pipeline Execution

---

### 🧩 Step 1: Feature Extraction from Labeled Data

- **Script:** `process_data.py`  
- **Input:** `./data/keypointlabel/keypoints_with_labels_<id>.csv` for IDs 1, 2, 3, 5  
- **Output:** `features_continuous_unfiltered.csv`  
- Extracts 70+ handcrafted features (motion, geometry, asymmetry, temporal-frequency)

---

### ⚙️ Step 2: Train Hybrid Model (Bi-LSTM + Transformer)

- **Script:** `train_hybrid_tuned.py`  
- **Output:** `saved_models/hybrid_tuned/`  
  - `best_model_fold_<id>.keras`, `scaler_fold_<id>.joblib`, `label_encoder.joblib`, and `feature_cols.json`

---

### ⚙️ Step 3: Train Bi-LSTM Model

- **Script:** `train_lstm_tuned.py`  
- **Output:** `saved_models/lstm_tuned/`  
  - `best_model_fold_<id>.keras`, `scaler_fold_<id>.joblib`, `label_encoder.joblib`, and `feature_cols.json`

---

### 📊 Step 4: LOSO Ensemble Evaluation

- **Script:** `ensemble_loso.py`  
- **Input:** trained models from step 2 and 3  
- **Output:** prints accuracy, F1-score, and per-fold classification reports

---

### 🏁 Step 5: Final Training – LSTM Model on All Data

- **Script:** `train_final_lstm_tuned_model.py`  
- **Output:** `final_lstm_tuned_model_artifacts/`  
  - Includes full model, scaler, label encoder, and selected features

---

### 🏁 Step 6: Final Training – Hybrid Model on All Data

- **Script:** `train_final_hybrid_model.py`  
- **Output:** `final_hybrid_model_artifacts/`  
  - Similar structure, includes Transformer block for long-range dependencies

---

### 🧪 Step 7: Feature Extraction from Test Set

- **Script:** `process_data_test.py`  
- **Input:** `test data_keypoint.csv`  
- **Output:** `features_test.csv`  
- Same features as training, built from interpolated keypoints

---

### 📤 Step 8: Create Submission

- **Script:** `create_submission.py`  
- **Combines:** `final_lstm_tuned_model` + `final_hybrid_model` (soft voting: 52/48)  
- **Output:** `Binary_Phoenix_test.csv` with:
  - `participant_id`, `timestamp`, `predicted_label`
