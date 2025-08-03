import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Path Configuration ---
# Use pathlib to get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Define the paths to the model and scaler files
MODEL_PATH = BASE_DIR / 'advanced_car_price_model.pkl'
SCALER_PATH = BASE_DIR / 'scaler.pkl'

# --- Load Model and Scaler Safely ---
try:
    # Load the machine learning model
    model = joblib.load(MODEL_PATH)
    # Load the scaler object used for normalization
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("Error: Could not find model or scaler files.")
    st.info(f"Please ensure the following files are in the same directory as 'app.py':")
    st.info(f" - {MODEL_PATH.name}")
    st.info(f" - {SCALER_PATH.name}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading files: {e}")
    st.stop()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("### Enter your car details below to predict the resale price 💸")

# --- Input Fields ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    with col2:
        age = st.slider("Car Age (in years)", min_value=0, max_value=30, value=5)

    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    ownertype = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# --- Manual Encoding Function ---
def encode_inputs():
    """Encodes the user inputs into the format expected by the model."""
    # Initialize arrays for one-hot encoded features
    fuel = [0, 0, 0, 0]  # Diesel, Electric, LPG, Petrol
    if fuel_type == 'Diesel': fuel[0] = 1
    elif fuel_type == 'Electric': fuel[1] = 1
    elif fuel_type == 'LPG': fuel[2] = 1
    elif fuel_type == 'Petrol': fuel[3] = 1

    # Encode seller type
    seller = 0 if seller_type == 'Dealer' else (1 if seller_type == 'Individual' else 0)
    trustmark = 1 if seller_type == 'Trustmark Dealer' else 0
    
    # Encode transmission
    manual = 1 if transmission == 'Manual' else 0

    # Encode owner type
    owner = [0] * 4
    if ownertype == 'Second Owner': owner[0] = 1
    elif ownertype == 'Third Owner': owner[1] = 1
    elif ownertype == 'Fourth & Above Owner': owner[2] = 1
    elif ownertype == 'Test Drive Car': owner[3] = 1

    # Combine all features into a single list
    features = [kms_driven, age] + fuel[:3] + [seller, trustmark, manual] + owner

    return np.array(features).reshape(1, -1)

# --- Predict Button and Logic ---
if st.button("Predict Price"):
    try:
        # Get the encoded user input
        input_data = encode_inputs()
        
        # Scale the numerical features (kms_driven, age) using the pre-trained scaler
        scaled_input = input_data.copy()
        scaled_input[:, :2] = scaler.transform(scaled_input[:, :2])
        
        # Make the prediction using the loaded model
        prediction = model.predict(scaled_input)
        
        # Display the result in a success message
        st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")
