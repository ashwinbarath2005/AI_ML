import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Diabetes Risk Predictor")

# Input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success(" Low Risk of Diabetes")
