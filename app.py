from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/winemodel.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():





    #Values
    fixed_acidity = float(request.form["fixed_acidity"])
    volatile_acidity = float(request.form["volatile_acidity"])
    citric_acid = float(request.form["citric_acid"])
    residual_sugar = float(request.form["residual_sugar"])
    chlorides = float(request.form["chlorides"])
    free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
    total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
    density = float(request.form["density"])
    ph = float(request.form["ph"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])

    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol


    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    quality = round(prediction[0])

    if quality <= 3:
        rating = "Poor Wine"

    elif quality <= 5:
        rating = "Average Wine"
    elif quality <= 7:
        rating = "Good wine"

    else:
        rating = "Execellent Wine"


    return render_template(
        "result.html",
        quality=quality,
        rating=rating
    )

if __name__ == "__main__":
    app.run(debug=True)


