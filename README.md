# Emotion_Classification
## 📌 Project Overview
This system detects **emotions in speech or song audio** using engineered features from audio signals. It works with `.wav` files and predicts the expressed emotion such as:
- `neutral`,`happy`, `sad`, `angry`, `fearful`, `calm`, `surprised`

The model is trained on the **RAVDESS** dataset, which includes professional recordings labeled with emotional content.

---

## 🧪 Pre-processing & Feature Engineering
We apply consistent preprocessing across both training and test data using `librosa`:

### Audio Features Extracted:
- **Mel Spectrogram (mean)**
- **MFCCs**, **Delta** and **Delta-Delta** features
- **Chroma STFT** (pitch class)
- **Spectral Contrast** (frequency contrast)
- **Tonnetz** (harmonic content)
- **Zero-Crossing Rate**, **RMS Energy**
- **Gender** and **Intensity** inferred from file naming

All features are combined into a single flat array to feed into our ML and DL models.

---

## 🧠 Model Training Pipeline

We experimented with multiple models:

### 🔍 Classical ML Models:
- Random Forest
- Logistic Regression
- SVM (Linear & RBF)
- XGBoost (with hyperparameter tuning)

### 🤖 Deep Learning:
- Dense Neural Network (Keras Sequential API)
- Optimized using dropout layers and batch normalization

### 📉 Evaluation:
- Used `80/20` train-validation split
- Models trained on a cleaned dataset excluding low-performance emotion classes('digust','fearful')
- Accuracy and F1-Score used for final evaluation

---

## ✅ Results Summary
- **Initial accuracy**: ~76%
- **After full feature engineering**: **~81% accuracy**
- **Best performance**: Achieved with XGBoost and optimized MLP
- **Evaluation Metric**: Weighted F1-score and precision

---

## 🧰 Tech Stack
- Python (Jupyter Notebooks + `.py` scripts)
- `librosa` for audio feature extraction
- `scikit-learn`, `xgboost`, `keras` for model training
- `joblib` for model saving and loading
- `streamlit` for deployment as a web app

---


---
