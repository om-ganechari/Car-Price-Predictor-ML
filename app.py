import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🚗 Car Price Predictor (ML Model)")

# INPUTS: These must match the features used in training
company = st.selectbox("Select Car Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Audi'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
kms_driven = st.number_input("Kilometers Driven", min_value=0)
age = st.number_input("Car Age (in years)", min_value=0)
seats = st.selectbox("No. of Seats", [4, 5, 6, 7, 8])

# Manual Encoding (this must match what was done during training)
def encode_inputs():
    company_map = {'Maruti': 0, 'Hyundai': 1, 'Honda': 2, 'Toyota': 3, 'Ford': 4, 'BMW': 5, 'Audi': 6}
    fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
    transmission_map = {'Manual': 0, 'Automatic': 1}
    seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
    owner_map = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}

    return [
        company_map[company],
        fuel_map[fuel_type],
        transmission_map[transmission],
        seller_map[seller_type],
        owner_map[owner],
        kms_driven,
        age,
        seats
    ]

# Prediction
if st.button("Predict Car Price"):
    input_data = np.array(encode_inputs()).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    st.success(f"💰 Estimated Car Price: ₹ {result[0]:,.2f}")
