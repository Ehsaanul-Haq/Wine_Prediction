import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataset/winequality.csv")
print("Dataset loaded successfully")
print(data.head())

X=data[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "ph",
        "sulphates",
        "alcohol"

    ]
]

y=data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("Model trained successfully")

predictions = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R2 Score", r2)

joblib.dump(
    model,
    "model/winemodel.pkl"
)

joblib.dump(scaler, "model/scaler.pkl")

print("Model saved to: wine tasting/model/winemodel.pkl")
print("Scaler saved to: wine tasting/model/scaler.pkl ")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Dataset shape:", data.shape)
print("Training Completed!")
