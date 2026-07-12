"""
Limpeza, engenharia de atributos e preparação para modelagem (Fases 2, 3 e 4 do notebook).
"""
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas completas e imputa 'sqft_above' ausente pela mediana."""
    df = df.drop_duplicates()
    df["sqft_above"] = df["sqft_above"].fillna(df["sqft_above"].median())
    return df.reset_index(drop=True)


def cap_iqr(serie: pd.Series, fator: float = 1.5) -> pd.Series:
    """Winsorização por IQR: valores fora de [Q1 - fator*IQR, Q3 + fator*IQR]
    são limitados (capped) ao limite mais próximo, em vez de removidos."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf, limite_sup = q1 - fator * iqr, q3 + fator * iqr
    return serie.clip(limite_inf, limite_sup)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as colunas derivadas da Fase 3: ano_venda, idade_imovel,
    foi_reformado e preco_por_m2 (esta última só para leitura - nunca deve
    entrar como variável preditora, pois deriva diretamente de 'price')."""
    df = df.copy()
    df["ano_venda"] = pd.to_datetime(df["date"].str[:8], format="%Y%m%d").dt.year
    df["idade_imovel"] = df["ano_venda"] - df["yr_built"]
    df["foi_reformado"] = (df["yr_renovated"] > 0).astype(int)
    df["preco_por_m2"] = df["price"] / df["sqft_living"]
    return df


def select_final_columns(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Seleciona apenas as variáveis explicativas e o alvo: o recorte de
    colunas que efetivamente entra na modelagem (Fase 4)."""
    return df[feature_cols + [target_col]].copy()


def fit_zipcode_faixas(X: pd.DataFrame, y: pd.Series, n_faixas: int = 10) -> tuple[dict, str]:
    """Calcula o mapeamento zipcode -> faixa de preço a partir do preço médio
    de venda por zipcode em X/y (target encoding). Retorna o dicionário de
    mapeamento e a faixa mais comum, usada como resposta padrão para
    zipcodes sem correspondência quando o mapeamento for aplicado depois."""
    preco_medio_por_zip = y.groupby(X["zipcode"]).mean()
    faixa_zip = pd.qcut(
        preco_medio_por_zip, n_faixas, labels=[f"faixa_{i}" for i in range(n_faixas)], duplicates="drop"
    )
    mapa_zip = faixa_zip.to_dict()
    faixa_mais_comum = faixa_zip.mode()[0]
    return mapa_zip, faixa_mais_comum


def apply_zipcode_faixas(
    X: pd.DataFrame, mapa_zip: dict, faixa_mais_comum: str, colunas_referencia=None
) -> pd.DataFrame:
    """Aplica um mapeamento zipcode -> faixa já calculado (fit_zipcode_faixas)
    a um DataFrame, com one-hot encoding. Se colunas_referencia for informado,
    alinha as colunas resultantes a essa referência (preenchendo com 0 o que
    faltar), garantindo que dois conjuntos codificados separadamente fiquem
    com as mesmas colunas."""
    X = X.copy()
    X["zipcode_faixa"] = X["zipcode"].map(mapa_zip).fillna(faixa_mais_comum)
    X = pd.get_dummies(X.drop(columns="zipcode"), columns=["zipcode_faixa"], drop_first=True)
    if colunas_referencia is not None:
        X = X.reindex(columns=colunas_referencia, fill_value=0)
    return X


def encode_zipcode(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, n_faixas: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa 'zipcode' em faixas de preço médio de venda (target encoding
    calculado somente com o treino, para evitar vazamento de dados) e aplica
    one-hot encoding. Zipcodes do teste sem correspondência no treino recebem
    a faixa mais comum vista no treino."""
    mapa_zip, faixa_mais_comum = fit_zipcode_faixas(X_train, y_train, n_faixas)
    X_train_enc = apply_zipcode_faixas(X_train, mapa_zip, faixa_mais_comum)
    X_test_enc = apply_zipcode_faixas(X_test, mapa_zip, faixa_mais_comum, colunas_referencia=X_train_enc.columns)
    return X_train_enc, X_test_enc
