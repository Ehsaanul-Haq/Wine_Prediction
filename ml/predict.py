import pandas as pd
import joblib

model = joblib.load ("wine tasting/model/winemodel.pkl")

scaler = joblib.load("wine tasting/model/scaler.pkl")
print("Model and scaler loaded successfully")
wine = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

input_data = pd.DataFrame([wine])
print("Input data:")
print(input_data)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

print("Predicted wine Quality:", prediction[0])