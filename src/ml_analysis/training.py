import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib

def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict,
    test_size: float = 0.2,
    mlflow_tracking_uri: str = None,
    experiment_name: str = "Stock Prediction",
    run_name: str = "Default Run",
    save_scaler_path: Path = None
) -> mlflow.entities.Run:
    """モデルを学習し、MLflowで実験を追跡する"""

    # 特徴量とターゲットを分離
    X = df[feature_cols]
    y = df[target_col]

    # 訓練データと検証データに分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False) #時系列データなのでシャッフルしない

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # MLflow設定
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow run: {run.info.run_id}")
        # パラメータのロギング
        mlflow.log_params(model_params)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target", target_col)
        mlflow.log_param("test_size", test_size)

        # モデルの初期化と学習
        # TODO: Allow selection of different model types
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)

        # 予測と評価
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted') # Adjust average as needed

        # メトリクスのロギング
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")

        # モデルのロギング (scikit-learnフレーバーを使用)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model logged to MLflow artifact path: model")

        # スケーラーのロギング (もしあれば)
        if save_scaler_path and save_scaler_path.exists():
            mlflow.log_artifact(str(save_scaler_path), "preprocessing")
            print(f"Scaler logged from: {save_scaler_path}")
        
        # 特徴量リストもアーティファクトとして保存
        features_path = Path("features.txt")
        with open(features_path, "w") as f:
            f.write("\n".join(feature_cols))
        mlflow.log_artifact(str(features_path), "metadata")
        features_path.unlink() # Clean up local file
        print(f"Feature list logged.")

        print(f"MLflow run completed: {run.info.run_id}")
        return run 