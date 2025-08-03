import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('advanced_car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🚗 Car Price Predictor")

# User inputs
kms_driven = st.number_input("KMs Driven", value=50000)
age = st.slider("Car Age (in years)", 0, 30, 5)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

# Manual encoding (must match training)
fuel_diesel = 1 if fuel_type == 'Diesel' else 0
trans_manual = 1 if transmission_type == 'Manual' else 0

# Final input: only 4 features
input_data = np.array([[kms_driven, age, fuel_diesel, trans_manual]])
scaled_input = scaler.transform(input_data)

if st.button("Predict Price 💰"):
    prediction = model.predict(scaled_input)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
