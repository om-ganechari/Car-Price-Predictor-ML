import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("quikr_car.csv")

# Data cleaning
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

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train[['Kms_Driven', 'Age']] = scaler.fit_transform(X_train[['Kms_Driven', 'Age']])
X_test[['Kms_Driven', 'Age']] = scaler.transform(X_test[['Kms_Driven', 'Age']])

# Model training
rf = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gs = GridSearchCV(rf, params, cv=3, n_jobs=-1, scoring='r2')
gs.fit(X_train, y_train)

# Save best model
joblib.dump(gs.best_estimator_, 'advanced_car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate
y_pred = gs.best_estimator_.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("✅ Model and Scaler saved!")

