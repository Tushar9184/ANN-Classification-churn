import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔥",
    layout="centered"
)


st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 30px;
    border-radius: 20px;
}
h1, h2, h3 {
    text-align: center;
}
.stButton>button {
    width: 100%;
    border-radius: 15px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    padding: 10px;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}
.success {
    background-color: rgba(0, 255, 150, 0.15);
}
.danger {
    background-color: rgba(255, 0, 100, 0.15);
}
</style>
""", unsafe_allow_html=True)


model = load_model("churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)   


st.markdown("<h1>🔥 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-powered banking insights</h3>", unsafe_allow_html=True)
st.markdown("---")


credit_score = st.slider("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.selectbox("Gender", ["Female", "Male"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

if st.button("🚀 Predict Churn"):

    
    has_card = 1 if has_card == "Yes" else 0
    is_active = 1 if is_active == "Yes" else 0

    
    gender_encoded = label_encoder.transform([gender])[0]

    
    geo_france = 1 if geography == "France" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    
    input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography_France": geo_france,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain,
    "Gender": gender_encoded,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary
    }])

    input_df = input_df[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_df)

    
    churn_prob = model.predict(input_scaled)[0][0]
    stay_prob = 1 - churn_prob

    
    if churn_prob > 0.5:
        st.markdown(
            f"""
            <div class='result-box danger'>
             <b>High Churn Risk</b><br>
            Churn Probability: {churn_prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='result-box success'>
             <b>Customer Likely to Stay</b><br>
            Stay Probability: {stay_prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
