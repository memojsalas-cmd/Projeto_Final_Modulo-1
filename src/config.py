"""
Caminhos e parâmetros centrais do projeto, usados pelo notebook e pelos
módulos de src/.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT_DIR / "data" / "raw" / "kc_house_data.csv"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "kc_house_processed.csv"

FINAL_DIR = ROOT_DIR / "data" / "final"
FINAL_FILE = FINAL_DIR / "kc_house_final.csv"

FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

MODEL_DIR = ROOT_DIR / "models" / "v1"
MODEL_FILE = MODEL_DIR / "modelo_regressao_v1.pkl"
METRICS_FILE = MODEL_DIR / "metricas_v1.json"

TARGET_COL = "price"
FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15",
    "sqft_lot15", "idade_imovel", "foi_reformado", "lat", "long", "zipcode",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
