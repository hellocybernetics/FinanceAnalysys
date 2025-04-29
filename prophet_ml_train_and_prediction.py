import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.ensemble import RandomForestRegressor # Replaced by Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler # Prophet handles scaling internally if needed
# import joblib # No longer saving model/scaler locally in this script
import matplotlib.pyplot as plt
from prophet import Prophet

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data # Keep initial cleaning
# Feature engineering functions are not directly used for basic Prophet
# from src.ml_analysis.feature_engineering import (...)

def parse_args():
    parser = argparse.ArgumentParser(description='Basic Time Series Forecasting with Prophet using Walk-Forward Validation.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='5y', help='Data fetch period (e.g., 5y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d)')
    # parser.add_argument('--initial_train_years', type=int, default=2, help='Initial number of years for training before the first validation') # Removed for simple train-test split
    return parser.parse_args()

def create_model():
    """Initializes and configures the Prophet model."""
    model = Prophet() # Use default settings for simplicity initially
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5) # Add monthly seasonality
    model.add_country_holidays(country_name='US') # Add US holidays
    return model

def main():
    args = parse_args()
    local_artifact_dir = Path('result') / 'prophet_forecast' / args.symbol
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Fetching ---
    print(f"[1/5] Fetching data for {args.symbol} ({args.period}, {args.interval})...")
    df_raw, _ = fetch_stock_data(args.symbol, period=args.period, interval=args.interval)
    if df_raw is None or df_raw.empty:
        print(f"Failed to fetch data for {args.symbol}. Exiting.")
        return
    print(f"Data fetched. Shape: {df_raw.shape}")

    # --- 2. Data Preparation for Prophet ---
    print("\n[2/5] Preparing data for Prophet...")
    # Initial cleaning (e.g., handling NaNs if any)
    df_cleaned = clean_data(df_raw)
    
    # Prophet requires columns 'ds' (datetime) and 'y' (value to forecast)
    df_prophet = df_cleaned.reset_index() # Get datetime index as column
    # Ensure the index is datetime type
    df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']) 
    
    # Keep only necessary columns for Prophet
    df_prophet = df_prophet[['ds', 'y']]
    print(f"Data prepared for Prophet. Shape: {df_prophet.shape}")
    # print(df_prophet.head())

    # --- 3. Train-Test Split (80% Train, 20% Test) ---
    print(f"\n[3/5] Performing Train-Test Split (80% Train, 20% Test)...")

    split_index = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split_index]
    test_df = df_prophet.iloc[split_index:]

    train_start, train_end = train_df['ds'].min(), train_df['ds'].max()
    test_start, test_end = test_df['ds'].min(), test_df['ds'].max()

    print(f"Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(train_df)} samples)")
    print(f"Test Period:     {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({len(test_df)} samples)")

    # Train model on the training set
    print("Training model on 80% data...")
    model = create_model()
    # Disable verbose logging from cmdstanpy during fit
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    model.fit(train_df)

    # Predict on the test set
    print("Predicting on 20% test data...")
    future_df_test = test_df[['ds']].copy()
    test_forecast = model.predict(future_df_test)

    # Merge forecast with actual test data
    test_results = pd.merge(test_df, test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    test_results.set_index('ds', inplace=True) # Set index for plotting

    # --- 4. Overall Evaluation ---
    print("\n[4/5] Evaluating Test Set Performance...")

    y_true_test = test_results['y']
    y_pred_test = test_results['yhat']

    # Calculate overall metrics
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    test_r2 = r2_score(y_true_test, y_pred_test)

    print("--- Test Set Metrics ---")
    print(f"Overall RMSE: {test_rmse:.4f}")
    print(f"Overall MAE:  {test_mae:.4f}")
    print(f"Overall R2:   {test_r2:.4f}")

    # --- 5. Results Visualization and Final Forecasting ---
    print("\n[5/5] Visualizing Results and Forecasting Future...")

    # Plot 1: Test set forecast using model.plot() and overlaying actuals
    print("\nGenerating Test Set Forecast Plot (model.plot style)...")
    test_set_plot_path = local_artifact_dir / 'prophet_test_set_forecast.png'
    fig_test = model.plot(test_forecast) # Use the model trained on train_df
    ax = fig_test.gca()
    # Overlay actual test data points
    ax.plot(test_df['ds'], test_df['y'], 'r.', label='Actual Test Price')
    ax.set_title(f'Prophet Test Set Forecast vs Actual ({args.symbol})')
    ax.legend() # Update legend to include actuals
    fig_test.savefig(test_set_plot_path)
    print(f"Test set forecast plot saved to: {test_set_plot_path}")
    plt.close(fig_test) # Close the figure

    # Train final model on ALL data for future forecasting
    print("\nTraining Final Model on All Data and Forecasting Future...")
    final_model = create_model() # Use the centralized model creation function

    print("Training final model on all available data...")
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING) # Keep logging quiet
    final_model.fit(df_prophet) # Train on the entire dataset

    # Create future dataframe for 1 year
    future_dates = final_model.make_future_dataframe(periods=365)
    print(f"Predicting future 1 year from {df_prophet['ds'].max().strftime('%Y-%m-%d')}...")
    future_forecast = final_model.predict(future_dates)

    # Save future forecast
    future_forecast_path = local_artifact_dir / 'prophet_future_1y_forecast.csv'
    future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(future_forecast_path, index=False)
    print(f"Future forecast data saved to: {future_forecast_path}")

    # Plot 2: Future forecast from the final model
    print("\nGenerating future forecast plot...")
    fig_future = final_model.plot(future_forecast)
    plt.title(f'Prophet Future 1-Year Forecast ({args.symbol})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    future_plot_path = local_artifact_dir / 'prophet_future_1y_forecast.png'
    fig_future.savefig(future_plot_path) # Save the figure object returned by plot()
    print(f"Future forecast plot saved to: {future_plot_path}")
    plt.close()

    print("\nProphet forecasting workflow with Train-Test split and future forecast finished.")
    print(f"Results saved in: {local_artifact_dir}")

if __name__ == "__main__":
    main() 