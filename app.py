import streamlit as st

# Title of the application
st.title("Air Quality Index Prediction")

# Description
st.write("Enter pollutant levels to predict AQI.")

# Input fields for pollutants
pm25 = st.number_input("Enter PM2.5 (µg/m³)", min_value=0.0, format="%.2f")
pm10 = st.number_input("Enter PM10 (µg/m³)", min_value=0.0, format="%.2f")
no2 = st.number_input("Enter NO2 (ppb)", min_value=0.0, format="%.2f")
so2 = st.number_input("Enter SO2 (ppb)", min_value=0.0, format="%.2f")

# Dummy prediction function
def predict_aqi(pm25, pm10, no2, so2):
    # Simple formula (not accurate, just for example)
    return (pm25 * 0.5 + pm10 * 0.3 + no2 * 0.15 + so2 * 0.05)

# Prediction button
if st.button("Predict AQI"):
    aqi = predict_aqi(pm25, pm10, no2, so2)
    st.success(f"Predicted AQI: {aqi:.2f}")

# Footer
st.write("This is a simple AQI prediction model. Use real models for accuracy.")
