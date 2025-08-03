import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load and clean data
df = pd.read_csv("Car details v3.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Age'] = 2025 - df['year']
df.drop(['year', 'name'], axis=1, inplace=True)

# Rename columns for clarity
df.rename(columns={
    'selling_price': 'Selling_Price',
    'km_driven': 'Kms_Driven',
    'fuel': 'Fuel_Type',
    'seller_type': 'Seller_Type',
    'transmission': 'Transmission_Type',
    'owner': 'Owner_Type'
}, inplace=True)

# Select only required features
df = df[['Kms_Driven', 'Age', 'Fuel_Type', 'Transmission_Type', 'Selling_Price']]
df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission_Type'], drop_first=False)

# Ensure all 4 required features exist
for col in ['Fuel_Type_Diesel', 'Transmission_Type_Manual']:
    if col not in df.columns:
        df[col] = 0

# Final feature list for training
features = ['Kms_Driven', 'Age', 'Fuel_Type_Diesel', 'Transmission_Type_Manual']
X = df[features]
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Evaluate
y_pred = model.predict(X_test)
print("✅ Training complete.")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
