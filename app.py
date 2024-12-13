# -*- coding: utf-8 -*-
"""
Energy Prediction Project

Predicting electrical energy consumption using multiple regression models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import zipfile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
TRAIN_ZIP = 'train.zip'
TEST_ZIP = 'test.zip'
MODEL_PATH = 'final_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Functions
def load_data_from_zip(zip_path):
    """Load and combine all CSV files from a zip file into a single DataFrame."""
    data_frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv'):
                logging.info(f"Reading {file_name} from {zip_path}")
                with z.open(file_name) as f:
                    df = pd.read_csv(f)
                    data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def preprocess_data(data):
    """Preprocess the dataset by handling missing values and feature engineering."""
    if data.isnull().values.any():
        logging.warning("Missing values found. Imputing with column means.")
        data.fillna(data.mean(), inplace=True)

    if 'time' in data.columns:
        data['time'] = pd.to_numeric(data['time'], errors='coerce')

    return data

def split_and_scale_data(data, features, target, test_size=0.2, random_state=42):
    """Split the dataset and scale the features."""
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train multiple models and return a Voting Regressor."""
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    gb = GradientBoostingRegressor(random_state=42)
    svr = SVR()

    voting = VotingRegressor([('lr', lr), ('rf', rf), ('gb', gb), ('svr', svr)])
    voting.fit(X_train, y_train)
    return voting

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    logging.info(f"Model Evaluation - RMSE: {rmse:.3f}, R^2: {r2:.3f}")
    return rmse, r2

# Main script
if __name__ == "__main__":
    logging.info("Loading training data...")
    train_data = load_data_from_zip(TRAIN_ZIP)
    train_data = preprocess_data(train_data)

    logging.info("Loading testing data...")
    test_data = load_data_from_zip(TEST_ZIP)
    test_data = preprocess_data(test_data)

    features = ['time', 'input_voltage']
    target = 'el_power'

    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(train_data, features, target)

    logging.info("Training the model...")
    model = train_models(X_train, y_train)

    logging.info("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")

# Streamlit app
st.set_page_config(page_title="Energy Consumption Prediction", page_icon="⚡", layout="wide")

st.sidebar.header("How to Use")
st.sidebar.write("1. Enter the input values for voltage and hours.\n"
                 "2. The model will predict the electrical energy consumption.\n"
                 "3. Visualize trends in the dataset or predictions separately.")

st.title("⚡ Electrical Energy Consumption Prediction")
st.markdown("### Predict electrical energy consumption and visualize trends independently!")

st.header("User Input for Prediction")
input_voltage = st.number_input("Enter the input voltage (V)", min_value=0.0, step=0.1)

hour = st.number_input("Enter the number of hours", min_value=0.0, step=0.1)

if st.button("🔮 Predict for User Input"):
    input_data = np.array([[input_voltage, hour]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.subheader("Predicted Electrical Energy Consumption:")
    st.write(f"{prediction[0]:.2f} kW")

st.header("Visualizing Trends in the Dataset")

@st.cache_data
def load_training_data():
    zip_file_path = 'train.zip'
    data_frames = []

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
                    data_frames.append(df)

    if data_frames:
        train_data = pd.concat(data_frames, ignore_index=True)
        return train_data
    else:
        raise ValueError("No CSV files found in the ZIP file")

train_data = load_training_data()

plt.figure(figsize=(10, 5))
if 'time' in train_data.columns:
    train_data['time'] = pd.to_datetime(train_data['time'])
    plt.plot(train_data['time'], train_data['el_power'], marker='o', linestyle='-', alpha=0.6)
    plt.title("Electrical Power Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption (kWh)")
    plt.xticks(rotation=45)
    plt.grid()
    st.pyplot(plt)
else:
    st.write("Time column is missing. Showing alternative trends.")
    plt.scatter(train_data['input_voltage'], train_data['el_power'], alpha=0.6)
    plt.title("Power Consumption vs Input Voltage")
    plt.xlabel("Input Voltage (V)")
    plt.ylabel("Power Consumption (kW)")
    st.pyplot(plt)

st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
st.pyplot(plt)
