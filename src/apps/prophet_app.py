# src/apps/prophet_app.py
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging

# Import necessary functions from other modules
# from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data # Removed
from src.ml_analysis.preprocessing import clean_data

# Disable cmdstanpy logging noise
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# @st.cache_data # Removed cache decorator from here
def run_prophet_forecast(df_raw, symbol, forecast_days): # Updated function signature
    """Trains Prophet, makes predictions, evaluates, and returns plots/metrics based on provided data."""
    # st.write(f"Fetching data for {symbol}...") # Removed
    # df_raw, _ = fetch_stock_data(symbol, period=period, interval=interval) # Removed

    # Data is now passed as argument 'df_raw'
    if df_raw is None or df_raw.empty:
        st.error("Input DataFrame is empty or None.")
        return None, None, None

    st.write("Preparing data for Prophet...")
    try:
        # Use the passed df_raw
        df_cleaned = clean_data(df_raw.copy()) # Use copy
        df_prophet = df_cleaned.reset_index()
        df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet[['ds', 'y']]
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, None

    if df_prophet.empty or len(df_prophet) < 10:
        st.error("Not enough data points after preparation to perform forecast.")
        return None, None, None

    st.write("Performing Train-Test Split (80/20)...")
    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    if train_df.empty or test_df.empty:
        st.error("Train or Test dataset is empty after split. Need more data.")
        return None, None, None

    st.write("Training Prophet model on training data...")
    try:
        model = Prophet()
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_country_holidays(country_name='US') # Assuming US market - might need adjustment
        model.fit(train_df)
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None, None, None

    st.write("Predicting on test data...")
    try:
        test_forecast = model.predict(test_df[['ds']])
    except Exception as e:
        st.error(f"Error predicting on test set: {e}")
        return None, None, None

    # Evaluate Test Set Performance
    test_results = pd.merge(test_df, test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    test_rmse = np.sqrt(mean_squared_error(test_results['y'], test_results['yhat']))
    test_mae = mean_absolute_error(test_results['y'], test_results['yhat'])
    try:
        test_r2 = r2_score(test_results['y'], test_results['yhat'])
    except ValueError:
        test_r2 = np.nan # R2 can be undefined in some cases

    metrics = {
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R2": test_r2
    }

    st.write("Generating test set forecast plot...")
    try:
        fig_test = model.plot(test_forecast)
        ax = fig_test.gca()
        ax.plot(test_df['ds'], test_df['y'], 'r.', markersize=3, label='Actual Test Price')
        ax.set_title(f'Prophet Test Set Forecast vs Actual ({symbol})')
        ax.legend()
        # plt.close(fig_test) # No need to close here, Streamlit handles it
    except Exception as e:
        st.error(f"Error generating test plot: {e}")
        fig_test = None

    st.write("Training final model on all data and forecasting future...")
    try:
        final_model = Prophet()
        final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        final_model.add_country_holidays(country_name='US')
        final_model.fit(df_prophet) # Train on the entire dataset

        future_dates = final_model.make_future_dataframe(periods=forecast_days)
        future_forecast = final_model.predict(future_dates)
    except Exception as e:
        st.error(f"Error during final model training or future forecasting: {e}")
        return fig_test, None, metrics # Return test plot if it was generated

    st.write("Generating future forecast plot...")
    try:
        fig_future = final_model.plot(future_forecast)
        ax_future = fig_future.gca()
        ax_future.set_title(f'Prophet Future {forecast_days}-Day Forecast ({symbol})')
        ax_future.set_xlabel('Date')
        ax_future.set_ylabel('Predicted Price')
        # plt.close(fig_future)
    except Exception as e:
        st.error(f"Error generating future plot: {e}")
        fig_future = None

    return fig_test, fig_future, metrics 