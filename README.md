🚗 Car Price Predictor – ML Project
A Machine Learning model that accurately predicts the price of a car based on various features like year, fuel type, seller type, transmission, and more.
Deployed with Streamlit Cloud for a smooth, interactive experience.

---
📌 Table of Contents
🔍 Problem Statement

📂 Dataset

🧠 ML Workflow

📊 Results

🚀 Deployment

🛠️ Tech Stack

📸 Demo

👨‍💻 Author

---
🔍 Problem Statement
Car pricing in the resale market is influenced by multiple dynamic features. Manual prediction is inaccurate and inefficient.
Goal: Build a regression model to accurately predict car prices using historical car sales data.

----
📂 Dataset
Source: Provided by Internpe

Format: .csv file

Features include:

Year (of manufacturing)

Present_Price

Kms_Driven

Fuel_Type

Seller_Type

Transmission

Owner

Price (Target)

-----

🧠 ML Workflow
Data Preprocessing

Handled missing values

Label encoding of categorical features

Feature scaling

Model Building

Trained a RandomForestRegressor

Evaluated using MAE, RMSE, R² score

Serialization

Saved trained model as model.pkl using joblib

Deployment

Streamlit frontend for real-time prediction

Deployed on Streamlit Cloud

-----


🔧 File Structure for Deployment
CarPricePredictor/
├── car data.csv
├── model.pkl                    ← trained model file
├── app.py                       ← Streamlit app
├── requirements.txt             ← for deployment
└── README.md                    ← optional

-----

📊 Results
Metric	Value
MAE	₹XXXX
RMSE	₹XXXX
R² Score	0.XX

🔥 Model achieves strong performance and generalization on test data.

-------

🚀 Deployment
🌐 Live Streamlit App: Click to Try It

-----

📁 To Run Locally:

bash
Copy
Edit
git clone https://github.com/your-username/car-price-predictor-ml.git
cd car-price-predictor-ml
pip install -r requirements.txt
streamlit run app.py
🛠️ Tech Stack
Python 🐍

Pandas, NumPy

Scikit-learn

Streamlit

Joblib

GitHub

-------

👨‍💻 Author
Om Ganechari
🎓 Artificial Intelligence & Data Science

