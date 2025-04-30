import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
# from sklearn.model_selection import TimeSeriesSplit # Not used for basic split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Added r2_score
# from sklearn.preprocessing import StandardScaler # Not typically needed for SARIMA on log price
# import joblib # Not saving model locally in this script
import matplotlib.pyplot as plt
# from arch import arch_model # Removed GARCH import
from statsmodels.tsa.statespace.sarimax import SARIMAX # Import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Import ACF/PACF plots
# import pmdarima as pm # Removed pmdarima import

# Add project root to Python path
# プロジェクトルート（srcディレクトリの親）をsys.pathに追加
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data # Keep initial cleaning
# Feature engineering not used for basic SARIMA

def parse_args():
    # Update description for SARIMA
    parser = argparse.ArgumentParser(description='Time Series Forecasting with SARIMA model using Train-Test Split.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='5y', help='Data fetch period (e.g., 5y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d)')
    # SARIMA orders could be added as args later
    # parser.add_argument('--order_p', type=int, default=1)
    # parser.add_argument('--order_d', type=int, default=1)
    # parser.add_argument('--order_q', type=int, default=1)
    return parser.parse_args()

# Removed GARCH create_model function

def main():
    args = parse_args()
    # Change result directory name for SARIMA
    local_artifact_dir = Path('result') / 'sarima_forecast' / args.symbol
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Fetching --- (Keep as is)
    print(f"[1/8] Fetching data for {args.symbol} ({args.period}, {args.interval})...") # Adjusted step count
    df_raw, _ = fetch_stock_data(args.symbol, period=args.period, interval=args.interval)
    if df_raw is None or df_raw.empty:
        print(f"Failed to fetch data for {args.symbol}. Exiting.")
        return
    print(f"Data fetched. Shape: {df_raw.shape}")

    # --- 2. Data Preparation for SARIMA ---
    # Update section title and steps for log price
    print("\n[2/8] Preparing data for SARIMA (Calculating Log Price)...") # Adjusted step count
    # Initial cleaning
    df_cleaned = clean_data(df_raw)

    # Calculate log price
    # Ensure 'Close' is positive before taking log
    if (df_cleaned['Close'] <= 0).any():
        print("Warning: Non-positive 'Close' prices found. Removing affected rows.")
        df_cleaned = df_cleaned[df_cleaned['Close'] > 0]
    df_cleaned['log_Close'] = np.log(df_cleaned['Close'])

    # Remove GARCH returns calculation
    # df_cleaned['returns'] = np.log(df_cleaned['Close'] / df_cleaned['Close'].shift(1)) * 100
    # df_cleaned = df_cleaned.dropna()

    # Extract log price series
    log_price_series = df_cleaned['log_Close']

    print(f"Log price calculated. Shape: {log_price_series.shape}")
    # print(log_price_series.head())

    # --- 3. Train-Test Split (80% Train, 20% Test) ---
    print(f"\n[3/8] Performing Train-Test Split on Log Price (80% Train, 20% Test)...") # Adjusted step count

    # Split the log price series
    split_index = int(len(log_price_series) * 0.8)
    train_data = log_price_series.iloc[:split_index]
    test_data = log_price_series.iloc[split_index:]

    # Keep track of dates for plotting and original prices for evaluation
    train_dates = df_cleaned.index[:split_index]
    test_dates = df_cleaned.index[split_index:]
    test_prices = df_cleaned['Close'].iloc[split_index:] # Keep original test prices

    train_start, train_end = train_dates.min(), train_dates.max()
    test_start, test_end = test_dates.min(), test_dates.max()

    print(f"Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(train_data)} samples)")
    print(f"Test Period:     {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({len(test_data)} samples)")

    # --- 3b. ACF/PACF Analysis for Order Selection --- (New Step)
    print("\n[3b/8] Generating ACF/PACF plots for differenced training data...") # Adjusted step number
    diff_train_data = train_data.diff().dropna() # Calculate first difference

    # Plot ACF
    acf_plot_path = local_artifact_dir / 'acf_diff_train.png'
    fig_acf = plt.figure(figsize=(12, 6))
    try:
        plot_acf(diff_train_data, ax=fig_acf.gca(), lags=40) # Show 40 lags
        plt.title('ACF Plot of Differenced Training Data (log_Close, d=1)')
        plt.savefig(acf_plot_path)
        print(f"ACF plot saved to: {acf_plot_path}")
    except Exception as e:
        print(f"Could not generate ACF plot: {e}")
    finally:
        plt.close(fig_acf)

    # Plot PACF
    pacf_plot_path = local_artifact_dir / 'pacf_diff_train.png'
    fig_pacf = plt.figure(figsize=(12, 6))
    try:
        plot_pacf(diff_train_data, ax=fig_pacf.gca(), lags=40, method='ywm') # method='ywm' often preferred
        plt.title('PACF Plot of Differenced Training Data (log_Close, d=1)')
        plt.savefig(pacf_plot_path)
        print(f"PACF plot saved to: {pacf_plot_path}")
    except Exception as e:
        print(f"Could not generate PACF plot: {e}")
    finally:
        plt.close(fig_pacf)

    print("=> Examine ACF/PACF plots to determine appropriate non-seasonal orders (p, q). Lags around multiples of 5 might indicate seasonality.")

    # --- 4. Model Training (SARIMA) ---
    # Update section title and steps for SARIMA
    print("\n[4/8] Training SARIMA Model...") # Adjusted step number

    # Define the model orders.
    # !! Examine ACF/PACF plots generated in step 3b to refine these orders !!
    order = (1, 1, 1) # (p, d, q) - Initial guess, adjust based on PACF(p) and ACF(q)
    seasonal_order = (1, 0, 0, 5) # (P, D, Q, s) - Initial guess for weekly seasonality (s=5)

    print(f"Using SARIMA order={order}, seasonal_order={seasonal_order}")
    print("NOTE: These orders are initial guesses. Review ACF/PACF plots for potential improvements.")

    # Remove auto_arima related code if any
    # print("Running auto_arima to find best SARIMA order...")
    # auto_arima_model = pm.auto_arima(...)

    model = SARIMAX(train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    # Fit the model (keep existing try/except)
    try:
        print("Fitting SARIMA model...")
        model_fit = model.fit(disp=False)
        print("Model fitting complete.")
    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        print("Skipping evaluation and visualization.")
        return

    # Print model summary
    print(model_fit.summary())

    # --- 5. Forecasting and Evaluation on Test Set ---
    print("\n[5/8] Forecasting on Test Set, Inverse Transforming, and Evaluating...") # Adjusted step number

    # Forecast horizon is the length of the test set
    forecast_horizon = len(test_data)

    # Perform forecasting using statsmodels SARIMAXResults object
    try:
        forecast_obj = model_fit.get_forecast(steps=forecast_horizon)
        predicted_log_mean = forecast_obj.predicted_mean
        conf_int_log = forecast_obj.conf_int(alpha=0.05)
        lower_log_limit = conf_int_log.iloc[:, 0]
        upper_log_limit = conf_int_log.iloc[:, 1]
    except Exception as e:
        # Adjusted error message
        print(f"Error during SARIMA get_forecast: {e}")
        print("Skipping evaluation and visualization.")
        return

    # Inverse transform to original price scale
    predicted_mean_price = np.exp(predicted_log_mean)
    lower_price = np.exp(lower_log_limit)
    upper_price = np.exp(upper_log_limit)

    # Align index for comparison and plotting
    predicted_mean_price.index = test_dates
    lower_price.index = test_dates
    upper_price.index = test_dates

    # Actual test prices (already extracted)
    y_true_price = test_prices

    # Ensure lengths match before evaluation
    if len(y_true_price) != len(predicted_mean_price):
        print("Warning: Length mismatch between actual and predicted prices after forecasting.")
        # This shouldn't happen with get_forecast, but handle just in case
        min_len = min(len(y_true_price), len(predicted_mean_price))
        y_true_price = y_true_price[:min_len]
        predicted_mean_price = predicted_mean_price[:min_len]
        lower_price = lower_price[:min_len]
        upper_price = upper_price[:min_len]
        test_dates_aligned = test_dates[:min_len]
    else:
        test_dates_aligned = test_dates


    # Calculate evaluation metrics on the price scale
    test_rmse = np.sqrt(mean_squared_error(y_true_price, predicted_mean_price))
    test_mae = mean_absolute_error(y_true_price, predicted_mean_price)
    test_r2 = r2_score(y_true_price, predicted_mean_price)

    print("\n--- Test Set Price Metrics ---")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"R2:   {test_r2:.4f}")


    # --- 6. Results Visualization and Saving ---
    print("\n[6/8] Visualizing Results and Saving Forecasts...") # Adjusted step number

    # Plot 1: Test Set Actual vs Predicted Price with Confidence Interval
    print("\nGenerating Test Set Price Forecast Plot...")
    price_plot_path = local_artifact_dir / 'sarima_test_price_forecast.png'
    plt.figure(figsize=(12, 6))
    plt.plot(df_cleaned.index, df_cleaned['Close'], label='Historical Price', color='grey', alpha=0.7) # Plot full history lightly
    plt.plot(test_dates_aligned, y_true_price, label='Actual Test Price', color='blue')
    plt.plot(test_dates_aligned, predicted_mean_price, label='Predicted Test Price (SARIMA)', color='red', linestyle='--')
    plt.fill_between(test_dates_aligned, lower_price, upper_price, color='pink', alpha=0.5, label='95% Confidence Interval')
    plt.title(f'SARIMA Test Set: Actual vs Predicted Price ({args.symbol})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # Limit x-axis to focus on the test period + some history for context
    # plot_start_date = test_dates_aligned.min() - pd.Timedelta(days=90) # Show 90 days before test start
    # plt.xlim(left=plot_start_date) # Removed xlim to show full history
    plt.savefig(price_plot_path)
    print(f"Test price forecast plot saved to: {price_plot_path}")
    plt.close()

    # Remove GARCH volatility plot

    # --- 7. Final Model Training and Future Forecasting ---
    print("\n[7/8] Training Final Model on All Data and Forecasting Future...") # Adjusted step number

    # Train final model on ALL log price data with the same selected orders
    print(f"Training final SARIMA model (order={order}, seasonal={seasonal_order}) on all available data...")
    # Remove auto_arima call for final model
    # print("Running auto_arima on ALL data for final model...")
    # final_model_fit = pm.auto_arima(...)

    final_model = SARIMAX(log_price_series, # Use full data
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    try:
        final_model_fit = final_model.fit(disp=False)
        print("Final model fitting complete.")
        # print(final_model_fit.summary()) # Optional: print summary of final model
    except Exception as e:
        print(f"Error fitting final SARIMA model: {e}")
        # ... error handling ...
        return

    # Forecast future periods using the final fitted statsmodels model
    future_steps = 365
    print(f"Predicting future {future_steps} days with final model...")
    try:
        # Use get_forecast for statsmodels
        future_forecast_obj = final_model_fit.get_forecast(steps=future_steps)
        future_log_mean = future_forecast_obj.predicted_mean
        future_conf_int_log = future_forecast_obj.conf_int(alpha=0.05)
        future_lower_log = future_conf_int_log.iloc[:, 0]
        future_upper_log = future_conf_int_log.iloc[:, 1]
    except Exception as e:
        # Adjusted error message
        print(f"Error during final model get_forecast: {e}")
        # ... error handling ...
    else:
        # Inverse transform future forecast
        future_mean_price = np.exp(future_log_mean)
        future_lower_price = np.exp(future_lower_log)
        future_upper_price = np.exp(future_upper_log)
        # Create future dates index
        last_date = log_price_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq=df_cleaned.index.freq)
        future_mean_price.index = future_dates # Apply index to the Series
        future_lower_price.index = future_dates
        future_upper_price.index = future_dates

    # Save future forecast data
    if not future_mean_price.empty:
        print("\nSaving future forecast to CSV...")
        future_forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_mean_price,
            'Lower_CI_95': future_lower_price,
            'Upper_CI_95': future_upper_price
        })
        future_forecast_csv_path = local_artifact_dir / 'sarima_future_forecast.csv'
        future_forecast_df.to_csv(future_forecast_csv_path, index=False)
        print(f"Future forecast data saved to: {future_forecast_csv_path}")


    # Plot 2: Future Forecast Plot
    if not future_mean_price.empty:
        print("\nGenerating future forecast plot...")
        future_plot_path = local_artifact_dir / 'sarima_future_forecast.png'
        plt.figure(figsize=(12, 6))
        plt.plot(df_cleaned.index, df_cleaned['Close'], label='Historical Price') # Plot historical data
        plt.plot(future_dates, future_mean_price, label='Future Predicted Price (SARIMA)', color='red', linestyle='--')
        plt.fill_between(future_dates, future_lower_price, future_upper_price, color='pink', alpha=0.5, label='95% Confidence Interval')
        plt.title(f'SARIMA Future Forecast ({args.symbol}) - {future_steps} Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        # Optional: Limit x-axis to show recent history + forecast
        # plot_future_start_date = df_cleaned.index[-1] - pd.Timedelta(days=180) # Show last 180 days history
        # plt.xlim(left=plot_future_start_date) # Removed xlim to show full history
        plt.savefig(future_plot_path)
        print(f"Future forecast plot saved to: {future_plot_path}")
        plt.close()


    # Save test set forecasts to CSV
    print("\n[8/8] Saving Test Set and Future Forecasts...") # Adjusted step number
    print("\nSaving test set forecasts to CSV...")
    test_forecast_df = pd.DataFrame({
        'Date': test_dates_aligned,
        'Actual_Price': y_true_price,
        'Predicted_Price': predicted_mean_price,
        'Lower_CI_95': lower_price,
        'Upper_CI_95': upper_price
    })
    test_forecast_csv_path = local_artifact_dir / 'sarima_test_forecasts.csv'
    test_forecast_df.to_csv(test_forecast_csv_path, index=False)
    print(f"Test set forecast data saved to: {test_forecast_csv_path}")


    # Removed GARCH forecast saving

    print("\nSARIMA forecasting workflow finished.")
    print(f"Results saved in: {local_artifact_dir}")

if __name__ == "__main__":
    main() 