from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


def dummy_predict(features):
   
    price, delivery_days, past_returns = features
    risk = 0.1 * price + 0.2 * delivery_days + 0.5 * past_returns
    risk_score = min(risk / 20, 1.0)  # Normalize risk to [0,1]
    return risk_score

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    price = float(request.form['price'])
    delivery_days = float(request.form['delivery_days'])
    past_returns = float(request.form['past_returns'])

    features = [price, delivery_days, past_returns]
    prediction = dummy_predict(features)
    
    return render_template('index.html', prediction=f"Predicted Return Risk: {prediction:.2%}")

if __name__ == '__main__':
    app.run(debug=True)
