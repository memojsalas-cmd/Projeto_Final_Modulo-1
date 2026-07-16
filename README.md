# Regressão Linear — Previsão do Preço de Imóveis

Projeto didático de ciência de dados que demonstra o pipeline de uma regressão linear: EDA, limpeza de dados, feature engineering, preparação para modelagem, modelagem e avaliação.

**Base de dados:** King County House Sales (Seattle, EUA) — 21.613 vendas de imóveis (2014–2015)  
**Modelo:** Regressão linear (`LinearRegression` do scikit-learn) sobre 23 variáveis (features físicas do imóvel + faixa de preço do zipcode, com escalonamento via `StandardScaler`), prevendo `price` diretamente em USD  
**R² (teste):** ~0,757 · **MAE (teste):** ~US$ 108.991

---

## Pipeline

| Fase | Descrição | Resultado |
|------|-----------|-----------|
| **Extração** | Download do dataset público | `data/raw/kc_house_data.csv` |
| **1 — EDA** | Distribuição de `price`, dispersão das variáveis com o alvo, mapa de correlação | figuras em `outputs/figures/` |
| **2 — Limpeza** | Remove duplicatas completas, imputa `sqft_above` ausente pela mediana, trata outlier de `bedrooms` (33 quartos) com winsorização por IQR (`clean_data`, `cap_iqr` em `src/features.py`) |
| **3 — Feature engineering** | Cria `ano_venda`, `idade_imovel`, `foi_reformado`; calcula `preco_por_m2` só para leitura (não usamos como preditora, pois deriva de `price`) (`add_features` em `src/features.py`); dataset resultante é salvo em disco (`save_processed_data` em `src/dataset.py`) | `data/processed/kc_house_processed.csv` |
| **4 — Preparação para modelagem** | Seleciona as 18 variáveis explicativas/features + variável-alvo (`select_final_columns`) e salva o recorte em na pasta data/final (`save_final_data`); a modelagem recarrega esse arquivo do zero (`load_final_data`) antes de continuar. Split treino/teste (80/20) *antes* do encoding; `zipcode` agrupado em 10 faixas de preço médio + one-hot encoding(`encode_zipcode` em `src/features.py`); remoção de colunas redundantes (`sqft_above`, `sqft_living15`, `sqft_lot15`) via análise de VIF; `StandardScaler` | `data/final/kc_house_final.csv` → matriz de 23 features escalonadas |
| **5 — Modelagem** | `LinearRegression` e `KNeighborsRegressor` treinados nas features escalonadas; comparação exploratória de valores de *k* e diagnóstico de overfitting (treino x teste) (`src/modeling/train.py`) | modelos avaliados |
| **6 — Avaliação e versionamento** | MAE/MSE/RMSE/R² calculados no split 80/20 (estimativa honesta de erro); gráficos de real x previsto e resíduos; **retreino final do modelo com 100% dos dados** (treino + teste, já que não há mais necessidade de reservar uma fatia para teste) antes de salvar; modelo retreinado e métricas do split salvos como v1 | `models/v1/`, `outputs/figures/` |

---

## Estrutura do projeto

```
├── data/
│   ├── raw/                        
│   ├── processed/                  
│   └── final/                      
│
├── models/
│   └── v1/
│       ├── modelo_v1.pkl.pkl 
│       └── metricas_v1.json        
│
├── notebooks/
│   └── dataview_precos.ipynb      
│
├── outputs/
│   └── figures/                    
│
├── src/
│   ├── config.py                   
│   ├── dataset.py                  
│   ├── features.py                 
│   ├── plots.py                    
│   └── modeling/
│       └── train.py                
│
└── requirements.txt
```

---

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Abrir o notebook
jupyter notebook notebooks/dataview_precos.ipynb
```

---

## Melhorias futuras

- **Empacotar o pré-processamento com o modelo:** hoje `models/v1/modelo_regressao_v1.pkl` salva só o `LinearRegression` treinado (retreinado com 100% dos dados na Fase 6). O `StandardScaler` e o mapeamento de zipcode→faixa (`fit_zipcode_faixas`/`apply_zipcode_faixas` em `src/features.py`) usados nesse retreino final não são salvos junto, eles existem apenas em memória durante a execução do notebook. Ou seja, para usar o modelo e prever em dados novos seria necessário recalcular esse pré-processamento manualmente nos dados novos, reproduzindo exatamente os mesmos passos. O ideal é migrar para um `sklearn.Pipeline`/`ColumnTransformer` que inclua encoding, escalonamento e modelo num único objeto salvo, permitindo carregar o pipeline completo e prever diretamente a partir de dados novos.
