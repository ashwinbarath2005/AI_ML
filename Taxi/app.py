from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open(r"C:\Users\Ashwinbarath\Desktop\taxi\data\model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Get form data
    priceperweek = float(request.form['priceperweek'])
    population = float(request.form['population'])
    monthlyincome = float(request.form['monthlyincome'])
    averageparkingpermonth = float(request.form['averageparkingpermonth'])

    # Create input DataFrame
    input_data = pd.DataFrame([[priceperweek, population, monthlyincome, averageparkingpermonth]],
                              columns=["priceperweek", "population", "monthlyincome", "averageparkingpermonth"])

    prediction = model.predict(input_data)[0]

    return render_template("result.html", prediction=round(prediction))

if __name__ == "__main__":
    app.run(debug=True)
