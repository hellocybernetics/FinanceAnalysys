import pandas as pd
import talib

# Potentially reuse existing calculation logic if suitable
# from src.analysis.technical.calc_technical import calculate_technical_indicators

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を特徴量として追加する"""
    # TODO: Implement technical feature generation (e.g., Moving Averages, RSI, MACD)
    try:
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        # Add more indicators as needed
        print("Technical features created.")
    except Exception as e:
        print(f"Error calculating TALib indicators: {e}. Ensure TALib is installed and data is sufficient.")
    return df

def create_lagged_features(df: pd.DataFrame, lags: list[int] = [1, 3, 5]) -> pd.DataFrame:
    """ラグ特徴量を追加する"""
    # TODO: Select features to lag
    feature_to_lag = 'Close' # Example
    if feature_to_lag in df.columns:
        for lag in lags:
            df[f'{feature_to_lag}_lag_{lag}'] = df[feature_to_lag].shift(lag)
        print(f"Lagged features created for '{feature_to_lag}' with lags: {lags}")
    else:
        print(f"Feature '{feature_to_lag}' not found for lagging.")
    return df

def create_target_variable(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """予測対象のターゲット変数を生成する (例: N日後のリターン)"""
    # Example: Predict next day return (shifted back for alignment with current features)
    df['target'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    # Example: Binary target (1 if price increased, 0 otherwise)
    # df['target_binary'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    print(f"Target variable created (predicting {horizon}-step ahead).")
    return df 