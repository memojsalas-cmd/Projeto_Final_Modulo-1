"""
Treino, avaliação e persistência de modelos de regressão (Fases 5 e 6).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR, MODEL_FILE, METRICS_FILE


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> KNeighborsRegressor:
    """Treina um KNN Regressor para comparar com a regressão linear."""
    n_neighbors = max(1, min(int(n_neighbors), len(X_train)))
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y_true) -> dict:
    """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_neighbors: int = 5,
) -> dict:
    """Compara regressão linear e KNN e retorna métricas de treino e teste."""
    lr_model = train_model(X_train, y_train)
    knn_model = train_knn_model(X_train, y_train, n_neighbors=n_neighbors)

    comparison = {
        "Regressão Linear": {
            "treino": evaluate_model(lr_model, X_train, y_train),
            "teste": evaluate_model(lr_model, X_test, y_test),
        },
        "KNN": {
            "treino": evaluate_model(knn_model, X_train, y_train),
            "teste": evaluate_model(knn_model, X_test, y_test),
        },
    }

    melhor_modelo = max(
        comparison,
        key=lambda nome: comparison[nome]["teste"]["R2"],
    )

    return {
        "comparacao": comparison,
        "melhor_modelo": melhor_modelo,
        "metricas_melhor_modelo": comparison[melhor_modelo]["teste"],
    }


def save_model(model) -> None:
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
