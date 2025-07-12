import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌾 Crop Recommendation System")
st.markdown("Enter soil and environmental conditions to predict the best crop to grow.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorous (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)  # If you have a saved scaler, use that instead
        prediction = model.predict(scaled)[0]
        st.success(f"🌱 Recommended Crop: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Error: {e}")
