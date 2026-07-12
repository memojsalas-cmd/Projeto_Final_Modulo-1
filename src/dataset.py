"""
Carregamento do dataset bruto de vendas de imóveis, além da leitura/escrita
dos datasets intermediário (processed) e pronto para modelagem (final).
"""
import pandas as pd

from src.config import FINAL_DIR, FINAL_FILE, PROCESSED_DIR, PROCESSED_FILE, RAW_FILE


def load_raw_data() -> pd.DataFrame:
    """Lê o CSV bruto em data/raw/ e devolve um DataFrame."""
    return pd.read_csv(RAW_FILE)


def save_processed_data(df: pd.DataFrame) -> None:
    """Salva em data/processed/ o dataset já limpo e com as features derivadas (Fases 2 e 3)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Dataset processado salvo em: {PROCESSED_FILE}")


def load_processed_data() -> pd.DataFrame:
    """Lê o dataset processado salvo em data/processed/."""
    return pd.read_csv(PROCESSED_FILE)


def save_final_data(df: pd.DataFrame) -> None:
    """Salva em data/final/ apenas as colunas usadas na modelagem (Fase 4)."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_FILE, index=False)
    print(f"Dataset final salvo em: {FINAL_FILE}")


def load_final_data() -> pd.DataFrame:
    """Lê o dataset final salvo em data/final/, pronto para a etapa de modelagem."""
    return pd.read_csv(FINAL_FILE)
