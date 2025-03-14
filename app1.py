import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import plotly.express as px
import matplotlib.pyplot as plt

# Load trained model
try:
    with open("aqi_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None
    st.error("⚠️ Model not found! Please upload 'aqi_model.pkl'.")

# Fetch live AQI data from OpenWeatherMap API
def fetch_live_aqi(city):
    API_KEY = "your_api_key_here"  # Replace with a valid API key
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=28.61&lon=77.23&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        aqi = data["list"][0]["main"]["aqi"]
        return aqi
    return None

# AQI category function
def get_aqi_category(aqi):
    categories = ["🟢 Good", "🟡 Moderate", "🟠 Unhealthy for Sensitive Groups", "🔴 Unhealthy", "🟣 Very Unhealthy", "⚫ Hazardous"]
    return categories[min(int(aqi // 50), 5)]

# Streamlit UI
title = "🌍 Air Quality Index (AQI) Prediction"
st.title(title)
st.write("🔢 Enter pollutant levels to predict AQI and compare with live data.")

# User input
pm25 = st.number_input("🌫️ PM2.5 (µg/m³)", min_value=0.0, format="%.2f")
pm10 = st.number_input("🌪️ PM10 (µg/m³)", min_value=0.0, format="%.2f")
no2 = st.number_input("🌫️ NO2 (ppb)", min_value=0.0, format="%.2f")
so2 = st.number_input("💨 SO2 (ppb)", min_value=0.0, format="%.2f")
co = st.number_input("🚗 CO (ppm)", min_value=0.0, format="%.2f")
o3 = st.number_input("🌍 O3 (ppb)", min_value=0.0, format="%.2f")
city = st.text_input("📍 Enter City Name (for live AQI):", "Delhi")

# Prediction
if st.button("🔍 Predict AQI"):
    if model:
        input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
        predicted_aqi = model.predict(input_data)[0]
        st.success(f"🌡️ **Predicted AQI: {predicted_aqi:.2f}** - {get_aqi_category(predicted_aqi)}")
        
        # Live AQI comparison
        live_aqi = fetch_live_aqi(city)
        if live_aqi:
            st.write(f"🌎 **Live AQI in {city}: {live_aqi} - {get_aqi_category(live_aqi)}**")
    else:
        st.error("⚠️ Model not found. Please upload 'aqi_model.pkl'.")

# Data visualization
if st.checkbox("📈 Show Pollutant Levels as Chart"):
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    values = [pm25, pm10, no2, so2, co, o3]
    fig = px.bar(x=pollutants, y=values, labels={'x': 'Pollutants', 'y': 'Concentration'}, title="Pollutant Levels")
    st.plotly_chart(fig)

# Time Series Visualization (if dataset is available)
if st.checkbox("📉 Show Historical AQI Trends"):
    df = pd.read_csv("air_quality_data.csv")  # Ensure dataset is available
    fig = px.line(df, x="Date", y="AQI", title="AQI Trends Over Time")
    st.plotly_chart(fig)

st.write("💡 **Tip:** If AQI is hazardous, limit outdoor activities!")
