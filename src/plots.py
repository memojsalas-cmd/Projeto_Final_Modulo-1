"""
Funções de plotagem usadas ao longo do notebook. Cada função gera uma
figura, salva em outputs/figures/ e a exibe com plt.show().
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR


def _savefig(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight", dpi=150)


def plot_histogram(serie: pd.Series, titulo: str, filename: str) -> None:
    """Histograma com curva de densidade (KDE) de uma variável numérica."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(serie, bins=60, kde=True, ax=ax, color="#2E7D32")
    ax.set_title(titulo)
    plt.tight_layout()
    _savefig(filename)
    plt.show()


def plot_boxplots(df: pd.DataFrame, cols: list[str], filename: str) -> None:
    """Um boxplot por coluna, lado a lado - útil para inspeção visual de outliers."""
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 4.5))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#9CCC65")
        ax.set_title(col)
    plt.tight_layout()
    _savefig(filename)
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str, filename: str) -> None:
    """Dispersão entre duas variáveis numéricas."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.25, s=15, ax=ax, color="#2E7D32")
    ax.set_title(f"{x} x {y}")
    plt.tight_layout()
    _savefig(filename)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, filename: str) -> None:
    """Mapa de calor de correlação de Pearson entre variáveis numéricas."""
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Correlação de Pearson entre variáveis numéricas")
    plt.tight_layout()
    _savefig(filename)
    plt.show()


def plot_observed_vs_predicted(y_true, y_pred, titulo: str, filename: str) -> None:
    """Valores reais x valores previstos - quanto mais perto da diagonal, melhor."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=15, color="#2E7D32")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#C62828", linewidth=1.5, label="Ajuste perfeito")
    ax.set_xlabel("Preço real (US$)")
    ax.set_ylabel("Preço previsto (US$)")
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    _savefig(filename)
    plt.show()


def plot_residuals(y_true, y_pred, titulo: str, filename: str) -> pd.Series:
    """Resíduos (real - previsto) versus valores previstos. Devolve os resíduos calculados."""
    residuos = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuos, alpha=0.7, color="#9CCC65")
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Valor previsto (US$)")
    ax.set_ylabel("Resíduo (real - previsto, em US$)")
    ax.set_title(titulo)
    plt.tight_layout()
    _savefig(filename)
    plt.show()
    return residuos
