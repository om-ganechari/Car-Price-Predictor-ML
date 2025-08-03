import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load and clean data
df = pd.read_csv("quikr_car.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Age'] = 2025 - df['year']
df.drop(['year', 'name'], axis=1, inplace=True)

# Rename columns
df.rename(columns={
    'selling_price': 'Selling_Price',
    'km_driven': 'Kms_Driven',
    'fuel': 'Fuel_Type',
    'seller_type': 'Seller_Type',
    'transmission': 'Transmission_Type',
    'owner': 'Owner_Type'
}, inplace=True)

# Keep only the features we need
df = df[['Kms_Driven', 'Age', 'Fuel_Type', 'Transmission_Type', 'Selling_Price']]

# One-hot encode and drop unused categories
df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission_Type'], drop_first=False)
df.drop(['Fuel_Type_Petrol', 'Transmission_Type_Automatic'], axis=1, errors='ignore', inplace=True)

# Add missing dummy columns if needed
for col in ['Fuel_Type_Diesel', 'Transmission_Type_Manual']:
    if col not in df.columns:
        df[col] = 0

# Final feature list
features = ['Kms_Driven', 'Age', 'Fuel_Type_Diesel', 'Transmission_Type_Manual']
X = df[features]
y = df['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (optional for RandomForest, kept for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
with open('advanced_car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Training complete.")
print("🔍 Feature order used for training:", features)
print("📈 R2 Score:", r2_score(y_test, y_pred))
print("📉 RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
