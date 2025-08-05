import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")

st.write("Fill in the details of the car to estimate its price.")

# Input fields
company = st.selectbox("Select Car Company", ["Toyota", "Honda", "BMW", "Hyundai", "Maruti", "Ford", "Tata", "Mahindra"])
car_model = st.text_input("Enter Car Model", placeholder="e.g. Swift, Fortuner")
year = st.number_input("Manufacturing Year", min_value=1995, max_value=2025, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
kms_driven = st.number_input("Kilometers Driven", min_value=0)

# Predict button
if st.button("Predict Price"):
    if car_model.strip() == "":
        st.error("Please enter a valid car model name.")
    else:
        # Prepare the input data
        input_df = pd.DataFrame({
            "name": [car_model],
            "company": [company],
            "year": [year],
            "kms_driven": [kms_driven],
            "fuel_type": [fuel_type]
        })

        # Apply the same preprocessing as training
        input_df_encoded = pd.get_dummies(input_df)
        
        # Match columns with training data
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in input_df_encoded:
                input_df_encoded[col] = 0
        input_df_encoded = input_df_encoded[expected_cols]

        # Scale input
        input_scaled = scaler.transform(input_df_encoded)

        # Predict
        predicted_price = model.predict(input_scaled)[0]
        st.success(f"Estimated Car Price: ₹ {predicted_price:,.2f}")
