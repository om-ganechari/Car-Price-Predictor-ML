import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('advanced_car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("### Enter your car details below to predict the resale price 💸")

# Input fields
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
age = st.slider("Car Age (in years)", min_value=0, max_value=30, value=5)

fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
ownertype = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# Manual encoding
def encode_inputs():
    fuel = [0, 0, 0, 0]  # Diesel, Electric, LPG, Petrol
    if fuel_type == 'Diesel': fuel[0] = 1
    elif fuel_type == 'Electric': fuel[1] = 1
    elif fuel_type == 'LPG': fuel[2] = 1
    elif fuel_type == 'Petrol': fuel[3] = 1

    seller = 0 if seller_type == 'Dealer' else (1 if seller_type == 'Individual' else 0)
    trustmark = 1 if seller_type == 'Trustmark Dealer' else 0
    manual = 1 if transmission == 'Manual' else 0

    owner = [0]*4
    if ownertype == 'Second Owner': owner[0] = 1
    elif ownertype == 'Third Owner': owner[1] = 1
    elif ownertype == 'Fourth & Above Owner': owner[2] = 1
    elif ownertype == 'Test Drive Car': owner[3] = 1

    features = [kms_driven, age] + fuel[:3] + [seller, trustmark, manual] + owner
    return np.array(features).reshape(1, -1)

# Predict button
if st.button("Predict Price"):
    input_data = encode_inputs()
    input_data[:, :2] = scaler.transform(input_data[:, :2])  # Scale Kms and Age
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:,.2f}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")
