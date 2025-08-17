from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("movie_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        prediction = model.predict(final_features)[0]
        
        return render_template("index.html", prediction_text=f"Movie Prediction: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
