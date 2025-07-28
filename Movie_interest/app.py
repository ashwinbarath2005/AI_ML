from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model1 = pickle.load(open(r'C:/Users/Ashwinbarath/Desktop/movie/data/movie_interest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# The form to input values
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from HTML form
    val1 = float(request.form['feature1'])
    val2 = float(request.form['feature2'])
    val3 = float(request.form['feature3'])
    val4 = float(request.form['feature4'])

    # Format input data for prediction (two user inputs)
    input_data = [[val1, val2], [val3, val4]]

    # Make predictions
    predictions = model1.predict(input_data)

    return render_template('result.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)
