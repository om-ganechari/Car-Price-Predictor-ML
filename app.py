import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and transformers
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🚗 Car Price Predictor")
st.markdown("Predict the **price of a used car** based on key features.")

with st.form("input_form"):
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=100)
    fuel = st.selectbox("Fuel Type", label_encoders['Fuel_Type'].classes_)
    seller_type = st.selectbox("Seller Type", label_encoders['Seller_Type'].classes_)
    transmission = st.selectbox("Transmission", label_encoders['Transmission'].classes_)
    owner = st.selectbox("Owner Type", label_encoders['Owner'].classes_)
    submit = st.form_submit_button("Predict Price")

if submit:
    fuel_encoded = label_encoders['Fuel_Type'].transform([fuel])[0]
    seller_encoded = label_encoders['Seller_Type'].transform([seller_type])[0]
    trans_encoded = label_encoders['Transmission'].transform([transmission])[0]
    owner_encoded = label_encoders['Owner'].transform([owner])[0]

    input_features = np.array([[year, kms_driven, fuel_encoded, seller_encoded, trans_encoded, owner_encoded]])
    input_scaled = scaler.transform(input_features)
    price = model.predict(input_scaled)[0]

    st.success(f"💰 Estimated Car Price: ₹{price:,.2f}")
