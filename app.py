import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")

st.title("🚘 Advanced Car Price Predictor App")
st.markdown("**Enter the details of the used car to predict its estimated selling price.**")

# Function to encode inputs
def encode_inputs():
    # Collect user inputs
    year = st.number_input("Year of Purchase", 1990, 2025, 2018)
    kms_driven = st.number_input("Kilometers Driven", 0, 1000000, 30000)
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    # Age
    age = 2025 - year

    # Encoding
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    fuel_petrol = 1 if fuel == 'Petrol' else 0

    seller_individual = 1 if seller_type == 'Individual' else 0
    seller_trustmark = 1 if seller_type == 'Trustmark Dealer' else 0

    trans_manual = 1 if transmission == 'Manual' else 0

    owner_2nd = 1 if owner == 'Second Owner' else 0
    owner_3rd = 1 if owner == 'Third Owner' else 0
    owner_4plus = 1 if owner == 'Fourth & Above Owner' else 0
    owner_test = 1 if owner == 'Test Drive Car' else 0

    # Order matters!
    data = [
        kms_driven,     # index 0
        age,            # index 1
        fuel_diesel,    # index 2
        fuel_petrol,    # index 3
        seller_individual,  # index 4
        seller_trustmark,   # index 5
        trans_manual,       # index 6
        owner_2nd,          # index 7
        owner_3rd,          # index 8
        owner_4plus,        # index 9
        owner_test          # index 10
    ]
    return data

# Main prediction
if st.button("Predict Selling Price 💰"):
    input_data = encode_inputs()

    # Extract scaler features (match your training!)
    kms_driven = input_data[0]
    age = input_data[1]
    fuel_diesel = input_data[2]
    trans_manual = input_data[6]

    # Scale those 4 features
    scaled_features = scaler.transform([[kms_driven, age, fuel_diesel, trans_manual]])[0]

    # Replace them back into input_data
    input_data[0] = scaled_features[0]  # scaled kms_driven
    input_data[1] = scaled_features[1]  # scaled age
    input_data[2] = scaled_features[2]  # scaled fuel_diesel
    input_data[6] = scaled_features[3]  # scaled trans_manual

    # Predict
    final_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(final_input)

    st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:,.2f}")


