# 🔥 Customer Churn Prediction App (ANN + Streamlit)

An end-to-end **Machine Learning + Deep Learning web application** that predicts whether a bank customer is likely to churn, built using an **Artificial Neural Network (ANN)** and deployed with **Streamlit**.

This project demonstrates **real-world ML deployment practices**, including preprocessing consistency, model serialization, and production-safe inference.

---

## 🚀 Live Demo

👉 _(Add your Streamlit Cloud URL here once deployed)_

---

## 📌 Problem Statement

Customer churn is a major challenge for banks and subscription-based businesses.  
Predicting churn in advance helps companies:

- Retain high-value customers
- Reduce revenue loss
- Design targeted retention strategies

This project uses historical customer data to **predict churn probability** using a trained ANN model.

---

## 🧠 Solution Overview

- Trained an **Artificial Neural Network (ANN)** for binary classification (Churn / No Churn)
- Applied **feature engineering & scaling** during training
- Saved preprocessing artifacts to ensure **training–inference consistency**
- Built an **interactive Streamlit web app** for real-time predictions
- Deployed with **production-safe practices** (caching, validation, error handling)

---

## 🧩 Tech Stack

### 🔹 Machine Learning / Deep Learning

- Python
- TensorFlow / Keras
- Artificial Neural Network (ANN)

### 🔹 Data Processing

- Pandas
- NumPy
- Scikit-learn
  - StandardScaler
  - LabelEncoder
  - One-Hot Encoding

### 🔹 Web & Deployment

- Streamlit
- Streamlit Cloud
- Git & GitHub

---

## 📊 Features

- 📈 Predicts **churn probability** (not just yes/no)
- 🎛️ Interactive UI with sliders and dropdowns
- 🌍 One-Hot Encoding for Geography (France, Germany, Spain)
- ⚖️ Scaled inputs using the **same scaler as training**
- 🧠 Cached model loading for faster performance
- 🛡️ Built-in validation & error handling
- 🎨 Clean, modern UI with custom styling

---

## 🗂️ Project Structure

ANN-Classification-churn/
│
├── app.py # Streamlit application
├── churn_model.h5 # Trained ANN model
├── scaler.pkl # Saved StandardScaler
├── label_encoder.pkl # Saved LabelEncoder
├── requirements.txt # Dependencies
├── runtime.txt # Python version for deployment
└── README.md # Project documentation

---

## ⚙️ How the Prediction Works

1. User enters customer details via the UI
2. Inputs are:
   - Label-encoded (Gender)
   - One-hot encoded (Geography)
   - Scaled using the **training scaler**
3. Processed data is passed to the ANN
4. Model outputs **churn probability**
5. App displays:
   - ✅ Customer Likely to Stay **OR**
   - ⚠️ High Churn Risk

---

## ▶️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Tushar9184/ANN-Classification-churn.git
cd ANN-Classification-churn
```

pip install -r requirements.txt
streamlit run app.py
