"""
Treino, avaliação e persistência de modelos de regressão (Fases 5 e 6).
"""
import json
import joblib
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor

from src.config import MODEL_DIR, MODEL_FILE, METRICS_FILE


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model



def cross_validate_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
    cv: int = 10,
):
    """Executa validação cruzada para o KNN e imprime um resumo estruturado."""
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    summary = {
        "R² médio": scores["test_r2"].mean(),
        "MAE médio": -scores["test_neg_mean_absolute_error"].mean(),
        "RMSE médio": -scores["test_neg_root_mean_squared_error"].mean(),
    }

    print("Resumo da validação cruzada do KNN:")
    print(f"  - R² médio: {summary['R² médio']:.4f}")
    print(f"  - MAE médio: {summary['MAE médio']:.4f}")
    print(f"  - RMSE médio: {summary['RMSE médio']:.4f}")

    return scores

def train_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "distance",
    cv: int = 5,
    grid_search: bool = True,
    cross_validation: bool = True,
) -> KNeighborsRegressor:
    """Treina um KNN Regressor com GridSearchCV e validação cruzada antes do fit final."""
    n_neighbors = max(1, min(int(n_neighbors), len(X_train)))
    base_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

    if grid_search:
        max_neighbors = max(1, min(31, len(X_train)))
        param_grid = {
            "n_neighbors": list(range(1, max_neighbors + 1)),
            "weights": ["uniform", "distance"],
        }

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
        print(f"Melhores hiperparâmetros do KNN: {best_params}")

        if cross_validation:
            print(f"Validação cruzada do KNN com {cv} folds:")
            cross_validate_knn(
                X_train,
                y_train,
                n_neighbors=best_params["n_neighbors"],
                weights=best_params["weights"],
                cv=cv,
            )

        return search.best_estimator_

    base_model.fit(X_train, y_train)

    if cross_validation:
        print(f"Validação cruzada do KNN com {cv} folds:")
        cross_validate_knn(
            X_train,
            y_train,
            n_neighbors=n_neighbors,
            weights=weights,
            cv=cv,
        )

    return base_model


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
