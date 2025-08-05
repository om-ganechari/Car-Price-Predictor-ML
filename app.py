import streamlit as st
import pandas as pd
import joblib

# Safe load function
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load {path}: {e}")
        return None

# Load model, scaler, and encoders
model = safe_load("model.pkl")
scaler = safe_load("scaler.pkl")
label_encoders = safe_load("label_encoders.pkl")

if not all([model, scaler, label_encoders]):
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor ML App")
st.markdown("### Enter Car Details to Predict Its Selling Price")

# Input fields
year = st.number_input("Car Purchase Year", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("Present Showroom Price (in Lakhs)", min_value=0.0, max_value=50.0, value=5.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", label_encoders['Fuel_Type'].classes_)
seller_type = st.selectbox("Seller Type", label_encoders['Seller_Type'].classes_)
transmission = st.selectbox("Transmission Type", label_encoders['Transmission'].classes_)

# Prediction logic
if st.button("Predict Selling Price"):
    try:
        # Encode categorical variables
        fuel_encoded = label_encoders['Fuel_Type'].transform([fuel_type])[0]
        seller_encoded = label_encoders['Seller_Type'].transform([seller_type])[0]
        transmission_encoded = label_encoders['Transmission'].transform([transmission])[0]

        # Feature vector
        input_data = [[year, present_price, kms_driven, fuel_encoded, seller_encoded, transmission_encoded, owner]]
        input_scaled = scaler.transform(input_data)

        # Predict
        predicted_price = model.predict(input_scaled)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

