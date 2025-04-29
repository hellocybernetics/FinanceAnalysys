from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys
import os
import yaml
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.ml_analysis.preprocessing import clean_data
from src.ml_analysis.feature_engineering import create_technical_features, create_lagged_features
from src.ml_analysis.prediction import load_and_predict

# Load environment variables (e.g., for MODEL_URI)
load_dotenv()

app = FastAPI(
    title="Stock Prediction API",
    description="API to predict stock price movements using a pre-trained ML model.",
    version="0.1.0"
)

# --- Configuration Loading ---
def load_api_config(config_path: str = 'config/api_config.yaml') -> dict:
    """Load API configuration (e.g., default model URI, feature params)."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded API configuration from: {config_path}")
        return config
    except Exception as e:
        print(f"Warning: Could not load API config from {config_path}. Using defaults or environment variables. Error: {e}")
        return {}

api_config = load_api_config()

# Determine Model URI: environment variable > config file > default
DEFAULT_MODEL_URI = "runs:/YOUR_DEFAULT_RUN_ID/model" # Replace with a sensible default or leave None
MODEL_URI = os.getenv("MODEL_URI", api_config.get("default_model_uri", DEFAULT_MODEL_URI))
if not MODEL_URI:
    print("Error: MODEL_URI not set via environment variable or api_config.yaml. API cannot function.")
    # Or raise an exception, or handle this case gracefully

# --- Pydantic Models (for request/response validation) ---
class PredictionResponse(BaseModel):
    symbol: str
    prediction_timestamp: str
    predictions: dict # Dict mapping timestamp to prediction details

# --- Helper Functions ---
def get_prediction_data(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """Fetch, preprocess, and engineer features for prediction."""
    df, _ = fetch_stock_data(symbol, period=period, interval=interval)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for symbol {symbol}")
    
    df_cleaned = clean_data(df)
    
    # Load feature engineering params from config (assuming api_config holds training config info or path)
    # This is fragile; ideally, feature params should be logged with the model in MLflow
    feature_eng_config = api_config.get('feature_engineering', {})
    lags = feature_eng_config.get('lags', [1, 3, 5])
    training_feature_cols = api_config.get('training', {}).get('feature_columns', []) # Need this for dropna
    
    if not training_feature_cols:
         print("Warning: training feature columns not found in config. Using fallback for dropna.")
         # Attempt to load from MLflow metadata if prediction function doesn't handle it

    df_features = create_technical_features(df_cleaned.copy())
    df_features = create_lagged_features(df_features, lags=lags)
    
    # Drop NaNs based on expected features *before* prediction
    # Using training_feature_cols helps ensure consistency
    if training_feature_cols:
        df_final = df_features.dropna(subset=training_feature_cols) 
    else:
        # Fallback: drop any row with any NaN - might lose recent data if features are NaN
        df_final = df_features.dropna()
        
    if df_final.empty:
         raise HTTPException(status_code=500, detail=f"No valid data remaining after feature engineering for {symbol}")
         
    return df_final

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API. Use the /predict/{symbol} endpoint."}

@app.post("/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock(symbol: str):
    """
    Predict stock movement for the given symbol using the configured ML model.
    Requires recent data fetching and feature engineering.
    """
    if not MODEL_URI:
         raise HTTPException(status_code=500, detail="ML Model URI is not configured.")
         
    try:
        print(f"Received prediction request for symbol: {symbol}")
        # 1. Get Data and Features
        data_for_prediction = get_prediction_data(symbol)
        print(f"Data prepared for prediction. Shape: {data_for_prediction.shape}")
        
        # 2. Load Model and Predict
        # We only need the latest prediction usually, but predict function might return all
        predictions_df = load_and_predict(data=data_for_prediction, model_uri=MODEL_URI)
        print(f"Predictions received from model. Shape: {predictions_df.shape}")
        
        # 3. Format Response
        # Convert timestamp index to string for JSON compatibility
        predictions_df.index = predictions_df.index.strftime('%Y-%m-%d %H:%M:%S')
        latest_prediction = predictions_df.iloc[-1:].to_dict('index') # Get the last prediction as a dict
        
        response = PredictionResponse(
            symbol=symbol,
            prediction_timestamp=list(latest_prediction.keys())[0], # Timestamp of the last prediction
            predictions=list(latest_prediction.values())[0] # Prediction details (prediction, probabilities)
        )
        return response

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except ValueError as ve:
        print(f"ValueError during prediction for {symbol}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error during prediction for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Optional: Add health check endpoint ---
@app.get("/health")
def health_check():
    # Basic health check: checks if API is running
    # More advanced checks could involve checking model loading or data source access
    return {"status": "ok"}

# --- Run Instructions (for local development) ---
# Run using: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 