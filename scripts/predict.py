import argparse
import pandas as pd
from pathlib import Path
import sys
import os
import yaml

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data # Potentially reuse scaling if needed, but predict usually needs scaler from training
from src.ml_analysis.feature_engineering import (
    create_technical_features,
    create_lagged_features,
    # Target variable is not needed for prediction
)
from src.ml_analysis.prediction import load_and_predict

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a trained model.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='1y', help='Data fetch period for prediction input (e.g., 1y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d)')
    parser.add_argument('--model_uri', type=str, required=True, help='MLflow model URI (e.g., runs:/<run_id>/model)')
    parser.add_argument('--output_file', type=str, default=None, help='Optional path to save predictions CSV')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Path to the *training* config file to get feature engineering parameters') # Need lags etc.

    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    config = load_config(args.config) # Load config to get feature params like lags

    # --- 1. Data Fetching ---
    print(f"Fetching data for {args.symbol}...")
    # Fetch enough data to calculate all necessary features (including lags)
    df, _ = fetch_stock_data(args.symbol, period=args.period, interval=args.interval)
    if df is None or df.empty:
        print(f"Failed to fetch data for {args.symbol}. Exiting.")
        return
    print(f"Data fetched successfully. Shape: {df.shape}")

    # --- 2. Preprocessing ---
    # Apply the same cleaning as in training
    print("Preprocessing data...")
    df_cleaned = clean_data(df)
    # Scaling is typically applied inside load_and_predict using the scaler logged with the model
    print("Data preprocessing complete.")

    # --- 3. Feature Engineering ---
    # Apply the same feature engineering steps as in training
    print("Creating features...")
    df_features = create_technical_features(df_cleaned.copy())
    df_features = create_lagged_features(df_features, lags=config['feature_engineering']['lags'])
    # Drop NaNs introduced *only* by feature engineering for prediction inputs
    # Keep the latest rows even if the target would have been NaN during training
    df_final = df_features.dropna(subset=config['training']['feature_columns']) 
    print(f"Feature engineering complete. Data shape for prediction: {df_final.shape}")

    if df_final.empty:
        print("No data left after feature engineering and NaN removal. Cannot predict.")
        return

    # --- 4. Prediction ---
    print(f"Loading model and making predictions using URI: {args.model_uri}...")
    try:
        predictions_df = load_and_predict(
            data=df_final, 
            model_uri=args.model_uri
        )
        print("Predictions generated:")
        print(predictions_df.tail()) # Print last few predictions

        # --- 5. Output Results ---
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(output_path)
            print(f"Predictions saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 