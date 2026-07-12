"""
Treino, avaliação e persistência do modelo de regressão linear (Fases 5 e 6).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR, MODEL_FILE, METRICS_FILE


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X, y_true) -> dict:
    """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def save_model(model: LinearRegression) -> None:
    """Salva o modelo treinado em models/v1/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Modelo salvo em: {MODEL_FILE}")


def save_metrics(metrics: dict) -> None:
    """Salva o dicionário de metadados/métricas em models/v1/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {METRICS_FILE}")
