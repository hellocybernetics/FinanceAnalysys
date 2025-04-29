import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理（欠損値処理など）を行う"""
    # TODO: Implement data cleaning logic (e.g., handling NaNs)
    df_cleaned = df.dropna() # Example: drop rows with NaNs
    print(f"Data cleaned. Shape before: {df.shape}, Shape after: {df_cleaned.shape}")
    return df_cleaned

def scale_features(df: pd.DataFrame, scaler_path: Path = None, fit_scaler: bool = False) -> tuple[pd.DataFrame, StandardScaler | None]:
    """特徴量をスケーリングする。fit_scaler=Trueの場合、新しいスケーラーを適合させ保存する。"""
    # TODO: Select features to scale
    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume'] # Example features
    features_to_scale = [f for f in features_to_scale if f in df.columns] # Ensure columns exist

    if not features_to_scale:
        print("No features selected for scaling.")
        return df, None

    scaler = None
    if fit_scaler:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        print(f"Features scaled and scaler fitted. Scaler saved to: {scaler_path}")
        if scaler_path:
             scaler_path.parent.mkdir(parents=True, exist_ok=True)
             joblib.dump(scaler, scaler_path)
    elif scaler_path and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        print(f"Features scaled using existing scaler: {scaler_path}")
    else:
        print("Warning: Scaling not performed. No scaler path provided or scaler file not found.")
        # Optionally, raise an error or proceed without scaling
        # raise ValueError("Scaler path required but not provided or file does not exist.")
        
    return df, scaler 