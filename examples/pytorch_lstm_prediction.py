import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm # For progress bar during training/prediction

# Add project root to Python path
# プロジェクトルート（srcディレクトリの親）をsys.pathに追加
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data
from src.ml_analysis.feature_engineering import create_technical_features

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting with PyTorch LSTM model.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='5y', help='Data fetch period (e.g., 5y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d)')
    parser.add_argument('--sequence_length', type=int, default=60, help='Number of past days for input sequence')
    parser.add_argument('--hidden_units', type=int, default=50, help='Number of units in LSTM hidden layers')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for LSTM layers')
    parser.add_argument('--future_steps', type=int, default=365, help='Number of days to forecast into the future')
    return parser.parse_args()

# --- PyTorch LSTM Model Definition ---
def create_lstm_model(input_size, hidden_units, num_layers, dropout_rate, output_size=1):
    """Creates the PyTorch LSTM model."""
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.hidden_units = hidden_units
            self.num_layers = num_layers
            # Add dropout to LSTM layers if more than one layer
            lstm_dropout = dropout_rate if num_layers > 1 else 0
            self.lstm = nn.LSTM(input_size, hidden_units, num_layers,
                                batch_first=True, dropout=lstm_dropout)
            # Optional: Add dropout layer after LSTM before Dense
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(hidden_units, output_size)

        def forward(self, x):
            # Initialize hidden and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)

            # We need to pass initial states to LSTM
            out, _ = self.lstm(x, (h0, c0))

            # Apply dropout to the output of the last LSTM layer sequence
            out = self.dropout(out[:, -1, :]) # Get output of the last time step

            # Pass through linear layer
            out = self.linear(out)
            return out
    return LSTMModel()

# --- Data Preparation Helper for LSTM ---
def create_sequences(data, sequence_length):
    """Converts time series data into sequences for LSTM."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length, 0] # Target is the first column (e.g., scaled 'Close')
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1) # Reshape y to [n_samples, 1]

def main():
    args = parse_args()
    # Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Result Directory ---
    local_artifact_dir = Path('result') / 'pytorch_lstm_forecast' / args.symbol
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Fetching ---
    print(f"\n[1/8] Fetching data for {args.symbol} ({args.period}, {args.interval})...")
    df_raw, _ = fetch_stock_data(args.symbol, period=args.period, interval=args.interval)
    if df_raw is None or df_raw.empty:
        print(f"Failed to fetch data for {args.symbol}. Exiting.")
        return
    print(f"Data fetched. Shape: {df_raw.shape}")

    # --- 2. Initial Cleaning ---
    print(f"\n[2/8] Cleaning data...")
    df_cleaned = clean_data(df_raw)

    # --- 3. Feature Engineering ---
    print(f"\n[3/8] Adding technical features...")
    df_featured = create_technical_features(df_cleaned.copy())
    # Drop rows with NaNs created by indicators
    initial_rows = len(df_featured)
    df_featured = df_featured.dropna()
    print(f"Dropped {initial_rows - len(df_featured)} rows with NaNs after feature engineering.")
    if df_featured.empty:
        print("No data left after feature engineering and NaN removal. Exiting.")
        return
    print(f"Data with features shape: {df_featured.shape}")

    # Define features to use (ensure 'Close' is first for target scaling/sequencing)
    features = ['Close'] + [col for col in df_featured.columns if col not in ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Close']]
    print(f"Using features: {features}")
    data_to_process = df_featured[features].values

    # --- 4. Train-Test Split ---
    print(f"\n[4/8] Performing Train-Test Split (80% Train, 20% Test)..." ) # Adjusted step number
    split_index = int(len(data_to_process) * 0.8)
    train_data_raw = data_to_process[:split_index]
    test_data_raw = data_to_process[split_index:]
    # Keep dates for plotting
    dates_full = df_featured.index
    train_dates_raw = dates_full[:split_index]
    test_dates_raw = dates_full[split_index:]
    # Keep original prices for evaluation
    original_prices = df_featured['Close'].values
    train_prices_orig = original_prices[:split_index]
    test_prices_orig = original_prices[split_index:]

    print(f"Train data shape (raw): {train_data_raw.shape}")
    print(f"Test data shape (raw): {test_data_raw.shape}")

    # --- 5. Data Scaling ---
    print(f"\n[5/8] Scaling data...")
    # Scale all features together
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)

    # Create a separate scaler for the 'Close' price (first column) for inverse transforming predictions
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    # Fit only on the training data's 'Close' price column
    scaler_price.fit(train_data_raw[:, 0].reshape(-1, 1))

    # --- 6. Sequence Creation and Tensor Conversion ---
    print(f"\n[6/8] Creating sequences and converting to Tensors...")
    X_train, y_train = create_sequences(train_data_scaled, args.sequence_length)
    X_test, y_test = create_sequences(test_data_scaled, args.sequence_length)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print(f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
    print(f"X_test shape: {X_test_tensor.shape}, y_test shape: {y_test_tensor.shape}")

    # Adjust test dates and original prices to match y_test length
    test_dates_aligned = test_dates_raw[args.sequence_length:]
    test_prices_aligned = test_prices_orig[args.sequence_length:]
    if len(test_dates_aligned) != len(y_test):
        print("Warning: Length mismatch between aligned test dates/prices and y_test. Check sequence creation.")
        # Adjust to minimum length if mismatch occurs (should not happen ideally)
        min_len_test = min(len(test_dates_aligned), len(y_test))
        test_dates_aligned = test_dates_aligned[:min_len_test]
        test_prices_aligned = test_prices_aligned[:min_len_test]
        y_test = y_test[:min_len_test]
        y_test_tensor = y_test_tensor[:min_len_test]
        X_test_tensor = X_test_tensor[:min_len_test]


    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Test loader is useful for batch evaluation, but can also predict on X_test_tensor directly
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 7. Model Training ---
    print(f"\n[7/8] Training LSTM Model...")
    num_features = X_train_tensor.shape[2]
    model = create_lstm_model(num_features, args.hidden_units, args.num_layers, args.dropout_rate).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    train_losses = []
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        # Use tqdm for progress bar on DataLoader
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_epoch_loss:.6f}")
        # Optional: Add validation loss calculation here using test_loader if desired

    print("Training finished.")

    # --- 8. Forecasting and Evaluation on Test Set (Recursive) ---
    print(f"\n[8/10] Forecasting on Test Set Recursively, Inverse Transforming, and Evaluating...") # Adjusted step number
    model.eval() # Set model to evaluation mode

    # Initialize the first input sequence from the test set
    initial_sequence_tensor = X_test_tensor[0].unsqueeze(0).to(device) # Add batch dimension
    current_batch_tensor = initial_sequence_tensor

    # List to store predictions (scaled)
    test_predictions_scaled_list = []

    # History for feature recalculation (actual scale)
    # Start with the last N days from the TRAINING data prices
    required_hist_len_test = args.sequence_length + 30 # Should match future forecast buffer
    # Ensure we take from the original prices BEFORE the split index
    price_history_for_features_test = list(original_prices[split_index - required_hist_len_test : split_index])

    print(f"Starting recursive forecasting for the test set ({len(y_test_tensor)} steps)...")
    # Recursive loop for the length of the test set targets
    for i in tqdm(range(len(y_test_tensor)), desc="Recursive Test Set Prediction"):
        with torch.no_grad():
            # 1. Predict next scaled price
            next_pred_scaled_tensor = model(current_batch_tensor)
            next_pred_scaled_np = next_pred_scaled_tensor.detach().cpu().numpy()
            test_predictions_scaled_list.append(next_pred_scaled_np.item()) # Store the scaled prediction

            # 2. Inverse transform predicted price to actual scale (for feature history)
            predicted_price_current_step = scaler_price.inverse_transform(next_pred_scaled_np.reshape(-1, 1)).flatten()[0]

            # 3. Update price history (actual scale)
            price_history_for_features_test.append(predicted_price_current_step)
            # Optional: Trim history length if it grows too large
            # price_history_for_features_test = price_history_for_features_test[-required_hist_len_test:]

            # 4. Recalculate features based on updated price history
            lookback_needed = 20 # Should match future forecast loop
            temp_df = pd.DataFrame({'Close': price_history_for_features_test[-lookback_needed:]})
            try:
                df_with_indicators = create_technical_features(temp_df)
                latest_indicators_unscaled = df_with_indicators.iloc[-1][features[1:]].values
                latest_features_unscaled = np.concatenate(([predicted_price_current_step], latest_indicators_unscaled))
            except Exception as e:
                print(f"\nWarning: Failed to recalculate indicators on test step {i+1}: {e}. Stopping test forecast.")
                # If features can't be calculated, stop the recursive forecast for test set
                # Fill remaining predictions with NaN or last valid prediction? For now, just stop.
                # Need to handle the length mismatch later during evaluation
                print("Stopping recursive test set forecast.")
                break # Exit the loop

            # 5. Scale the new features vector
            latest_features_unscaled_reshaped = latest_features_unscaled.reshape(1, -1)
            new_step_features_scaled = scaler.transform(latest_features_unscaled_reshaped)

            # 6. Update the input sequence tensor
            current_sequence_np = current_batch_tensor.squeeze(0).cpu().numpy()[1:] # Drop oldest step
            next_sequence_np = np.vstack((current_sequence_np, new_step_features_scaled)) # Append newest step
            current_batch_tensor = torch.tensor(next_sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Post-Loop Processing for Test Set ---
    print("Finished recursive test set forecast loop.")
    # Convert collected scaled predictions to NumPy array
    test_predictions_scaled_np = np.array(test_predictions_scaled_list).reshape(-1, 1)

    # Inverse transform the predictions to original price scale
    # This will only contain predictions for steps where the loop didn't break
    predicted_price = scaler_price.inverse_transform(test_predictions_scaled_np).flatten()

    # Actual prices - We need to align y_true_price with the number of successful predictions
    num_predictions_made = len(predicted_price)
    y_true_price = test_prices_aligned[:num_predictions_made]
    test_dates_eval = test_dates_aligned[:num_predictions_made]

    print(f"Number of test predictions made: {num_predictions_made}")
    if num_predictions_made < len(y_test_tensor):
        print(f"Warning: Test forecast stopped early. Evaluating only on {num_predictions_made} predictions.")

    # Calculate evaluation metrics (only if predictions were made)
    if num_predictions_made > 0:
        test_rmse = np.sqrt(mean_squared_error(y_true_price, predicted_price))
        test_mae = mean_absolute_error(y_true_price, predicted_price)
        test_r2 = r2_score(y_true_price, predicted_price)

        print("\n--- Test Set Price Metrics (Recursive) ---")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE:  {test_mae:.4f}")
        print(f"R2:   {test_r2:.4f}")
    else:
        print("\nNo test predictions were made. Skipping evaluation.")
        # Set dummy values or handle downstream
        test_rmse, test_mae, test_r2 = np.nan, np.nan, np.nan

    # Save test set forecasts (only if predictions were made)
    if num_predictions_made > 0:
        print("\nSaving recursive test set forecasts...")
        test_forecast_df = pd.DataFrame({
            'Date': test_dates_eval, # Use dates corresponding to predictions made
            'Actual_Price': y_true_price,
            'Predicted_Price': predicted_price
        })
        test_forecast_csv_path = local_artifact_dir / 'pytorch_lstm_test_recursive_forecasts.csv' # New filename
        test_forecast_df.to_csv(test_forecast_csv_path, index=False)
        print(f"Recursive test set forecast data saved to: {test_forecast_csv_path}")
    else:
        print("\nNo recursive test set forecasts to save.")

    # --- 9. Results Visualization (Test Set - Recursive) ---
    print(f"\n[9/10] Visualizing Recursive Test Set Results...")
    price_plot_path = local_artifact_dir / 'pytorch_lstm_test_recursive_price_forecast.png' # New filename
    plt.figure(figsize=(14, 7))
    # Plotting historical context
    plot_hist_start_idx = max(0, split_index - args.sequence_length - 180)
    plt.plot(dates_full[plot_hist_start_idx:split_index+args.sequence_length], 
             original_prices[plot_hist_start_idx:split_index+args.sequence_length], 
             label='Historical Price', color='grey', alpha=0.7)
    # Plot actual prices for the period where predictions were made
    plt.plot(test_dates_eval, y_true_price, label='Actual Test Price', color='blue')
    # Plot predicted prices (recursive)
    if num_predictions_made > 0:
        plt.plot(test_dates_eval, predicted_price, label='Predicted Test Price (LSTM - Recursive)', color='red', linestyle='--')
    plt.title(f'PyTorch LSTM Recursive Test Set Forecast ({args.symbol})') # Updated title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(price_plot_path)
    print(f"Recursive test set forecast plot saved to: {price_plot_path}")
    plt.close()

    # --- 10. Final Model Training and Future Forecasting ---
    # Note: Retraining on full data is computationally expensive and requires careful state management.
    # Here we use the model trained on the initial 80% for future forecast.
    print(f"\n[10/10] Forecasting Future {args.future_steps} Days (Using Model Trained on 80% Data, Recalculating Features)...")

    model.eval() # Ensure model is in eval mode

    # Get the last sequence from the original SCALED data for the initial prediction
    last_sequence_scaled = scaler.transform(data_to_process)[-args.sequence_length:]
    # Convert to tensor and add batch dimension
    current_batch_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Keep track of predicted prices in their original scale
    future_predicted_price = []

    # Keep track of the price history (original scale) needed for feature calculation
    # Start with the last N days from the original data (N >= sequence_length + max_lookback_for_features)
    # Determine max lookback needed by features (e.g., SMA10 needs 10, RSI14 needs 14 -> max is 14)
    # For simplicity, let's take a bit more than sequence_length
    required_hist_len = args.sequence_length + 30 # A buffer for indicator calculation
    price_history_for_features = list(original_prices[-required_hist_len:])

    print(f"Starting recursive future forecasting with feature recalculation...")
    for i in tqdm(range(args.future_steps), desc="Forecasting Future"):
        with torch.no_grad():
            # 1. Predict next scaled price
            next_pred_scaled_tensor = model(current_batch_tensor)
            next_pred_scaled_np = next_pred_scaled_tensor.detach().cpu().numpy()

            # 2. Inverse transform predicted price to actual scale
            predicted_price_current_step = scaler_price.inverse_transform(next_pred_scaled_np.reshape(-1, 1)).flatten()[0]
            future_predicted_price.append(predicted_price_current_step)

            # 3. Update price history (actual scale) for feature calculation
            price_history_for_features.append(predicted_price_current_step)
            # Keep history length manageable if needed, but ensure enough for feature calc
            # price_history_for_features = price_history_for_features[-required_hist_len:]

            # 4. Recalculate features based on updated price history
            # Create temporary DataFrame
            lookback_needed = 20 # Example max lookback for SMA10, RSI14 etc.
            temp_df = pd.DataFrame({'Close': price_history_for_features[-lookback_needed:]})
            try:
                df_with_indicators = create_technical_features(temp_df)
                # Get the LATEST calculated indicators
                latest_indicators_unscaled = df_with_indicators.iloc[-1][features[1:]].values # Exclude 'Close'
                # Combine predicted price (actual scale) and latest indicators (actual scale)
                latest_features_unscaled = np.concatenate(([predicted_price_current_step], latest_indicators_unscaled))

            except Exception as e:
                print(f"\nWarning: Failed to recalculate indicators on step {i+1}: {e}. Using previous step's indicators.")
                # Fallback: Use the indicators from the previous step's input
                # This fallback is complex and potentially inaccurate. For stability, we stop forecasting.
                print("Stopping future forecast due to error in feature recalculation.")
                break # Stop the loop

            # --- Continue loop only if feature recalculation succeeded --- 
            # (The 'break' in except block handles the failure case)

            # 5. Scale the new features vector
            # Reshape to (1, num_features) for the scaler
            latest_features_unscaled_reshaped = latest_features_unscaled.reshape(1, -1)
            new_step_features_scaled = scaler.transform(latest_features_unscaled_reshaped)

            # 6. Update the input sequence tensor
            # Get current sequence as numpy array, drop oldest step
            current_sequence_np = current_batch_tensor.squeeze(0).cpu().numpy()[1:]
            # Append the new scaled features
            next_sequence_np = np.vstack((current_sequence_np, new_step_features_scaled))
            # Update current_batch_tensor for the next iteration
            current_batch_tensor = torch.tensor(next_sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Post-Loop Processing ---
    # Inverse transform (already done inside loop and stored in future_predicted_price)
    future_predicted_price = np.array(future_predicted_price).flatten()

    # Create future dates (remains the same)
    last_date = dates_full[-1]
    # Ensure future_dates matches the length of predictions actually made
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predicted_price))

    # Save future forecast data (check if predictions were generated)
    if len(future_predicted_price) > 0:
        print("\nSaving future forecast to CSV...")
        future_forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predicted_price
        })
        future_forecast_csv_path = local_artifact_dir / 'pytorch_lstm_future_forecast.csv'
        future_forecast_df.to_csv(future_forecast_csv_path, index=False)
        print(f"Future forecast data saved to: {future_forecast_csv_path}")

    # Plot Future Forecast (check if predictions were generated)
    if len(future_predicted_price) > 0:
        print("\nGenerating future forecast plot...")
        future_plot_path = local_artifact_dir / 'pytorch_lstm_future_forecast.png'
        plt.figure(figsize=(14, 7))
        # Plot historical data (e.g., last year)
        hist_plot_start_date = dates_full[-1] - pd.Timedelta(days=365)
        plt.plot(dates_full[dates_full >= hist_plot_start_date], 
                 original_prices[dates_full >= hist_plot_start_date],
                 label='Historical Price')
        plt.plot(future_dates, future_predicted_price, label=f'Future Predicted Price (LSTM) - {args.future_steps} Days', color='red', linestyle='--')
        plt.title(f'PyTorch LSTM Future Forecast ({args.symbol})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(future_plot_path)
        print(f"Future forecast plot saved to: {future_plot_path}")
        plt.close()

    print(f"\nPyTorch LSTM forecasting workflow finished.")
    print(f"Results saved in: {local_artifact_dir}")

if __name__ == "__main__":
    main() 