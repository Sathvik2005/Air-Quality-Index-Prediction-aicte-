# streamlit
#app 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the dataset
df = pd.read_csv('air quality data.csv')
# Preprocessing steps (same as your previous code)
# ...

# Feature & Target Selection
X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
y = df['AQI']

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regressor": SVR(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

# Streamlit application
st.title("AQI Prediction Model")

# Sidebar for selecting the model
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Fit and evaluate the selected model
model = models[model_name]
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train, train_pred))
RMSE_test = np.sqrt(mean_squared_error(y_test, test_pred))
r2_train = r2_score(y_train, train_pred)
r2_test = r2_score(y_test, test_pred)

# Display results
st.write(f"## {model_name} Results")
st.write(f"### RMSE Train Data: {RMSE_train}")
st.write(f"### RMSE Test Data: {RMSE_test}")
st.write(f"### R Squared value for Train: {r2_train}")
st.write(f"### R Squared value on Test: {r2_test}")

# Option to view dataset
if st.checkbox('Show dataset'):
    st.write(df)

# Visualizations
st.write("## Data Distribution")
st.write(sns.displot(df, x='AQI', color='red'))
st.pyplot()
