from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\Ashwinbarath\Desktop\student_performance\data\model_performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance_percentage'])
        grade = float(request.form['previous_grade_numeric'])

        data = [[study_hours, attendance, grade]]
        prediction = model.predict(data)

        performance_map = {0: 'Poor', 1: 'Average', 2: 'Good'}
        result = performance_map.get(prediction[0], "Unknown")
    except Exception as e:
        result = f"Error: {e}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
