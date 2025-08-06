import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("return_predictor.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Return Predictor", page_icon="📦")

st.title("📦 Product Return Risk Predictor")
st.markdown("Predict whether a product will likely be returned based on order details.")

# Input fields
category = st.number_input("Product Category (0 = Books, 1 = Clothing, 2 = Electronics)", min_value=0, max_value=2)
price = st.number_input("Product Price (₹)", min_value=0.0)
rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0, step=0.1)
delivery_time = st.number_input("Delivery Time (in days)", min_value=1)

# Predict button
if st.button("🔍 Predict Return Risk"):
    input_data = np.array([[category, price, rating, delivery_time]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ This product is likely to be RETURNED.")
    else:
        st.success("✅ This product is likely NOT RETURNED.")
