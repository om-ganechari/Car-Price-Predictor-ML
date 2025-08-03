import streamlit as st
import numpy as np
import pickle

# Load model and scaler (both must be in same folder)
with open('advanced_car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")

# User Inputs
kms_driven = st.number_input("KMs Driven", min_value=0, value=30000)
age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

# Manual encoding (must match training pipeline)
fuel_diesel = 1 if fuel_type == 'Diesel' else 0
trans_manual = 1 if transmission == 'Manual' else 0

if st.button("Predict Price"):
    # Create array for scaler
    arr = np.array([[kms_driven, age, fuel_diesel, trans_manual]])
    scaled = scaler.transform(arr)

    # Prediction
    pred = model.predict(scaled)[0]
    st.success(f"Estimated Price: ₹ {round(pred, 2)} Lakhs")
