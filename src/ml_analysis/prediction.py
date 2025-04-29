import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

def load_and_predict(data: pd.DataFrame, model_uri: str) -> pd.DataFrame:
    """MLflowからモデルと前処理オブジェクトをロードし、予測を行う"""
    
    print(f"Loading model from: {model_uri}")
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")

    # MLflow Runから関連するアーティファクトのパスを取得
    client = mlflow.tracking.MlflowClient()
    run_id = model_uri.split('/')[1] # Extract run_id from model_uri (e.g., runs:/RUN_ID/model)
    artifacts = client.list_artifacts(run_id)
    
    scaler = None
    scaler_artifact_path = None
    features = None
    features_artifact_path = None

    # スケーラーと特徴量リストのアーティファクトを探す
    for artifact in artifacts:
        if artifact.path.endswith('scaler.pkl') and artifact.path.startswith('preprocessing/'): # Assuming scaler is saved as scaler.pkl in preprocessing dir
             scaler_artifact_path = client.download_artifacts(run_id, artifact.path)
             print(f"Found scaler artifact: {artifact.path}")
        elif artifact.path == 'metadata/features.txt':
            features_artifact_path = client.download_artifacts(run_id, artifact.path)
            print(f"Found features artifact: {artifact.path}")

    # スケーラーのロード
    if scaler_artifact_path:
        scaler = joblib.load(scaler_artifact_path)
        print(f"Scaler loaded from {scaler_artifact_path}")
    else:
        print("Scaler artifact not found in the specified run.")

    # 特徴量リストのロード
    if features_artifact_path:
        with open(features_artifact_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        print(f"Features loaded from {features_artifact_path}: {features}")
    else:
        print("Features artifact not found in the specified run.")
        # If features not logged, we might need to infer or have them passed differently
        # For now, assume prediction fails if features are missing
        raise ValueError("Feature list artifact (metadata/features.txt) not found in the MLflow run.")

    # 入力データに必要な特徴量が存在するか確認
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features in input data: {missing_features}")
        
    # 予測に必要な特徴量のみを選択
    data_predict = data[features]

    # スケーリングの適用 (スケーラーが存在する場合)
    if scaler:
        # Get column names before scaling (as scaler transforms to numpy array)
        scaled_feature_names = scaler.get_feature_names_out() # Assumes scaler was fit on pandas DF
        data_predict[scaled_feature_names] = scaler.transform(data_predict[scaled_feature_names])
        print("Input data scaled using loaded scaler.")
    else:
        print("Proceeding without scaling as scaler was not loaded.")

    # 予測の実行
    predictions = loaded_model.predict(data_predict)
    # 予測確率の取得 (分類モデルの場合)
    try:
        probabilities = loaded_model.predict_proba(data_predict)
        print("Predictions and probabilities generated.")
        # 結果をDataFrameに格納 (元のindexを使用)
        result_df = pd.DataFrame({
            'prediction': predictions,
             # Add probabilities for each class
            **{f'probability_class_{i}': probabilities[:, i] for i in range(probabilities.shape[1])} 
        }, index=data.index)
    except AttributeError:
         # モデルが predict_proba を持たない場合 (例: 回帰)
         print("Predictions generated (predict_proba not available).")
         result_df = pd.DataFrame({'prediction': predictions}, index=data.index)

    return result_df 