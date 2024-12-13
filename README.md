# Energy Prediction Web App

## Overview
The **Energy Prediction Web App** is a machine learning-based platform designed to predict electrical energy consumption. It leverages a Voting Regressor model trained on historical data and provides users with real-time predictions and visualizations for energy usage trends. Built with Streamlit, the app is user-friendly and highly interactive.

## Features
- **Real-Time Prediction**: Input voltage and hours to predict energy consumption.
- **Data Visualizations**: Analyze trends in energy consumption with interactive charts.
- **Machine Learning Models**: Combines Linear Regression, Random Forest, Gradient Boosting, and SVR in a robust ensemble model.
- **Scalable Design**: Uses pre-trained models and scalers for consistent predictions.

## Usage Instructions
1. **Input Values**: Enter the required voltage and hours to predict energy consumption.
2. **Visualize Data**: Explore trends and correlations in the provided dataset.
3. **Real-Time Insights**: View predictions and insights directly on the app.

## Installation
1. Clone the repository:
   ```bash
   git clone (https://github.com/Keerthana-B-15/Energy_Prediction_WebApp.git)
   cd Energy_Prediction_WebApp
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py`: Main Streamlit application script.
- `train.zip` & `test.zip`: Datasets used for training and testing.
- `final_model.pkl`: Pre-trained machine learning model.
- `scaler.pkl`: StandardScaler for input data normalization.

## Technical Details
- **Programming Language**: Python
- **Frameworks/Libraries**: Streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn
- **Machine Learning**: Voting Regressor combining multiple regression techniques

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for enhancements or bug fixes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For queries, contact [Keerthana B](keerthanab610@gmail.com).
