import streamlit as st
import joblib
import pandas as pd
from datetime import date

# Load model and encoders
model = joblib.load("model.pkl")
le_product = joblib.load("product_encoder.pkl")
le_store = joblib.load("store_encoder.pkl")

# UI
st.title("Supermarket Demand Predictor")

product = st.selectbox("Select Product", le_product.classes_)
store = st.selectbox("Select Store", le_store.classes_)
promo = st.radio("Promotion Running?", ["Yes", "No"])
selected_date = st.date_input("Pick a Date", date.today())

# Predict
if st.button("Predict Demand"):
    day = pd.to_datetime(selected_date).dayofweek
    promo_val = 1 if promo == "Yes" else 0
    p = le_product.transform([product])[0]
    s = le_store.transform([store])[0]

    input_data = [[p, s, promo_val, day]]
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Demand: **{round(prediction)} units**")
