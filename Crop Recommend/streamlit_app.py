import streamlit as st
import numpy as np
import joblib

# Load your saved model
model = joblib.load('model.pkl')

# App title and description
st.title("🌾 Crop Prediction App")
st.markdown("Enter soil and environmental parameters to get the recommended crop.")

# Input form for features
N = st.number_input("Nitrogen content in soil (N)", min_value=0.0)
P = st.number_input("Phosphorus content in soil (P)", min_value=0.0)
K = st.number_input("Potassium content in soil (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

# Combine into input array
features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# Predict and show result
if st.button("Predict Crop"):
    prediction = model.predict(features)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")
