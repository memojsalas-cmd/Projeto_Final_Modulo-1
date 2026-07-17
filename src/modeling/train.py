"""
Treino, avaliação e persistência dos modelos (Fases 5 e 6).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from src.config import MODEL_DIR, MODEL_FILE, METRICS_FILE


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_knn_model(
    X_train: pd.DataFrame | tuple, y_train: pd.Series | None = None, n_neighbors: int = 30
) -> KNeighborsRegressor:
    
    """Treina um KNN Regressor para comparação com a Regressão Linear."""
    
    if y_train is None:
        if not isinstance(X_train, tuple) or len(X_train) != 2:
            raise TypeError("Informe X_train e y_train para treinar o modelo KNN.")
        X_train, y_train = X_train

    model = KNeighborsRegressor(n_neighbors=n_neighbors)
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

def evaluate_model2(model: KNeighborsRegressor, X, y_true) -> dict:
   """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
        
   y_pred = model.predict(X)
   mse = mean_squared_error(y_true, y_pred)
   return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }   

def save_model(model, model_name: str) -> None:
    """ Salva o modelo campeão utilizando seu nome. """
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, MODEL_FILE)
    print(f"Modelo campeão ({model_name}) salvo em: {MODEL_FILE}")

def save_metrics(metrics: dict) -> None:
    """Salva o dicionário de metadados/métricas em models/v1/."""
   
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {METRICS_FILE}")
