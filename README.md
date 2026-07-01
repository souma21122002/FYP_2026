# 🌧️ Time-Series Rainfall Detection using BiLSTM

A deep learning model that predicts rainfall from historical hourly weather data using a stacked **Bidirectional LSTM (BiLSTM)** architecture. This is Part 3 of a larger final-year project on rainfall prediction (*Rainfall Prediction from Cloud Images using Deep Learning*, IIEST Shibpur), focusing on the **time-series weather-based approach**.

---

## 📌 Overview

Unlike image-based rainfall classification, this approach treats rainfall detection as a **sequence learning problem** — using the past 24 hours of weather observations to predict whether it will rain in the next hour.

| | |
|---|---|
| **Task** | Binary Classification (Rain / No Rain) |
| **Model** | 3-layer Bidirectional LSTM |
| **Input** | 24-hour lookback window of weather features |
| **Dataset** | Kaggle Historical Hourly Weather Data (Vancouver) |
| **Test Accuracy** | 83.34% |
| **F1-Score** | 0.8517 |
| **ROC-AUC** | 0.8830 |

---

## 📂 Dataset

**Source:** [Historical Hourly Weather Data](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data) (Kaggle)

- **City:** Vancouver
- **Time Range:** 2012 – 2017
- **Total Samples:** ~45,000 hourly records
- **Raw Features:** Humidity, Pressure, Temperature, Wind Speed, Wind Direction, Weather Description (text)

### Label Creation
Since the raw dataset only has a text `description` field, a binary `rain` label was derived by matching against a set of rain-related keywords:

```
['rain', 'drizzle', 'shower', 'thunderstorm', 'sleet', 'mist',
 'freezing rain', 'heavy intensity rain', 'light rain',
 'moderate rain', 'very heavy rain']
```

**Class distribution (before balancing):**
- No Rain: ~67.2%
- Rain: ~31.1%

---

## 🛠️ Preprocessing & Feature Engineering

1. **Missing value handling** — rows with nulls dropped (`dropna()`)
2. **Class balancing** — minority (Rain) class upsampled to match majority class
3. **Cyclical time encoding** — `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
4. **Lag features** — humidity & pressure at t-3, t-6, t-12, t-24 hours
5. **Rolling averages** — to smooth short-term noise
6. **Normalization** — Min-Max scaling applied to all features
7. **Sequence generation** — sliding window of **24 timesteps** to predict the label at t+1

```python
LOOKBACK = 24  # past 24 hours used to predict the next hour

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)
```

### Train / Val / Test Split
Time-ordered split (**no shuffling**, to preserve temporal structure) — **70 / 15 / 15**

| Split | Samples | Shape |
|---|---|---|
| Train | 39,151 | (39151, 24, n_features) |
| Validation | 8,389 | (8389, 24, n_features) |
| Test | 8,390 | (8390, 24, n_features) |

---

## 🧠 Model Architecture

A 3-block stacked **Bidirectional LSTM** built with the Keras Functional API:

```
Input (24 timesteps, n_features)
        │
        ▼
Bidirectional LSTM (128 units, return_sequences=True)
Dropout (0.2) + Recurrent Dropout (0.1)
Batch Normalization
        │
        ▼
Bidirectional LSTM (64 units, return_sequences=True)
Dropout (0.2) + Recurrent Dropout (0.1)
Batch Normalization
        │
        ▼
Bidirectional LSTM (32 units, return_sequences=False)
Dropout (0.2)
Batch Normalization
        │
        ▼
Dense (64, ReLU) → Dropout (0.3)
Dense (32, ReLU) → Dropout (0.2)
        │
        ▼
Dense (1, Sigmoid) → Rainfall Probability (0–1)
```

```python
def build_bilstm(n_timesteps, n_features):
    inputs = Input(shape=(n_timesteps, n_features))

    x = Bidirectional(LSTM(128, return_sequences=True,
                            dropout=0.2, recurrent_dropout=0.1))(inputs)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(64, return_sequences=True,
                            dropout=0.2, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, dropout=0.2))(x)
    x = BatchNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
```

**Why BiLSTM?** Bidirectional layers let the model learn from both forward (past → future) and backward (future → past) temporal patterns within each 24-hour window, capturing weather trends that a unidirectional LSTM might miss.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.0001) |
| Loss Function | Binary Crossentropy |
| Batch Size | 64 |
| Max Epochs | 60 (early stopped ~epoch 42) |
| Early Stopping | monitor=`val_auc`, patience=10, restore best weights |
| LR Scheduling | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |
| Checkpointing | Saves best weights by `val_auc` |

---

## 📊 Results

### Test Set Performance
| Metric | Score |
|---|---|
| Accuracy | 83.34% |
| F1-Score | 0.8517 (at optimal threshold 0.36) |
| ROC-AUC | 0.8830 |

### Classification Report
```
              precision    recall  f1-score   support
    No Rain       0.86      0.75      0.80      3830
       Rain       0.81      0.90      0.85      4560

    accuracy                           0.83      8390
   macro avg       0.84      0.82      0.83      8390
weighted avg       0.83      0.83      0.83      8390
```

### Confusion Matrix
| | Predicted: No Rain | Predicted: Rain |
|---|---|---|
| **Actual: No Rain** | 3032 (TN) | 798 (FP) |
| **Actual: Rain** | 600 (FN) | 3960 (TP) |

The model correctly identifies ~90% of actual rain events (recall), which is particularly valuable for early-warning use cases where missing a rain event is costlier than a false alarm.

---

## 🚀 Inference

A helper function is included to run predictions on new/recent weather data:

```python
def predict_rainfall(model, scaler, recent_data, lookback=24, threshold=0.5):
    """
    Predict rainfall from the most recent `lookback` hours of weather data.
    recent_data: DataFrame with the same feature columns used in training.
    """
    window = recent_data.iloc[-lookback:][ALL_FEATURES].values
    scaled = scaler.transform(window)
    seq = scaled.reshape(1, lookback, -1)

    prob = model.predict(seq, verbose=0)[0][0]
    label = 'RAIN' if prob >= threshold else 'NO RAIN'
    return prob
```

---

## 📁 Repository Structure

```
.
├── notebook.ipynb              # Full training & evaluation pipeline
├── rainfall_bilstm_final.h5    # Trained model weights
├── rainfall_scaler.pkl         # Fitted MinMaxScaler
├── rainfall_features.pkl       # List of feature columns used
└── README.md
```

### Loading the saved model
```python
import tensorflow as tf
import joblib

model  = tf.keras.models.load_model('rainfall_bilstm_final.h5')
scaler = joblib.load('rainfall_scaler.pkl')
features = joblib.load('rainfall_features.pkl')
```

---

## 🔮 Future Work

- Extend to multi-class rainfall **intensity** prediction (light / moderate / heavy) instead of binary rain/no-rain
- Multi-region validation beyond Vancouver
- Combine with image-based (satellite/ground-cloud) models for a multimodal fusion approach
- Real-time deployment with a live weather API feed

---

## 🧰 Tech Stack

`Python` · `TensorFlow / Keras` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## 👤 Author

**Soumadip Singha Mahapatra**
B.Tech Information Technology, IIEST Shibpur (2022ITB097)
Final Year Project — under the guidance of Prof. Santi P. Maity

---

*Part of a larger multi-approach rainfall prediction study covering ground-based multimodal cloud imaging, satellite image classification, and this time-series weather-based BiLSTM model.*
