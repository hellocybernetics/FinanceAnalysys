import argparse
import pandas as pd
from pathlib import Path
import sys
import os
import yaml

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data, scale_features
from src.ml_analysis.feature_engineering import (
    create_technical_features,
    create_lagged_features,
    create_target_variable
)
from src.ml_analysis.training import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a stock prediction model.')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='5y', help='Data fetch period (e.g., 5y)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d)')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Path to training configuration YAML file')
    # Add MLflow related arguments if needed (e.g., tracking URI, experiment name override)
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None, help='MLflow tracking URI (optional, defaults to local ./mlruns)')
    parser.add_argument('--experiment_name', type=str, default="Stock Prediction", help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None, help='MLflow run name (optional)')

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
    config = load_config(args.config)

    # --- 1. Data Fetching ---
    print(f"Fetching data for {args.symbol}...")
    df, _ = fetch_stock_data(args.symbol, period=args.period, interval=args.interval)
    if df is None or df.empty:
        print(f"Failed to fetch data for {args.symbol}. Exiting.")
        return
    print(f"Data fetched successfully. Shape: {df.shape}")

    # --- 2. Preprocessing ---
    print("Preprocessing data...")
    df_cleaned = clean_data(df)
    
    # Define scaler path
    output_dir = Path('result') / 'ml_artifacts' / args.symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / 'scaler.pkl'
    
    df_scaled, _ = scale_features(df_cleaned, scaler_path=scaler_path, fit_scaler=True)
    print("Data preprocessing complete.")

    # --- 3. Feature Engineering ---
    print("Creating features...")
    df_features = create_technical_features(df_scaled.copy()) # Use copy to avoid modifying df_scaled inplace if needed later
    df_features = create_lagged_features(df_features, lags=config['feature_engineering']['lags'])
    df_final = create_target_variable(df_features, horizon=config['feature_engineering']['target_horizon'])
    
    # Drop rows with NaNs introduced by feature engineering (lags, indicators, target shift)
    df_final = df_final.dropna()
    print(f"Feature engineering complete. Final data shape: {df_final.shape}")

    # --- 4. Model Training ---
    print("Starting model training...")
    feature_cols = config['training']['feature_columns']
    # Ensure only existing columns are used
    feature_cols = [col for col in feature_cols if col in df_final.columns]
    print(f"Using features: {feature_cols}")
    target_col = config['training']['target_column']
    model_params = config['model']['params']

    if target_col not in df_final.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return
        
    if not feature_cols:
        print("Error: No valid feature columns specified or found in the data.")
        return

    # Set run name if not provided
    run_name = args.run_name if args.run_name else f"{args.symbol}_{config['model']['type']}_training"

    train_model(
        df=df_final,
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=model_params,
        test_size=config['training']['test_split_ratio'],
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        run_name=run_name,
        save_scaler_path=scaler_path # Pass scaler path for logging
    )
    print("Model training finished.")

if __name__ == "__main__":
    main() 