## Precificação de Imóveis (King County, EUA)

--Sobre o Projeto  

Este repositório contém um Projeto didático cujo objetivo principal é resolver um problema preditivo de regressão: estimar o preço de venda (price) de imóveis, em dólares americanos (USD), comparando o desempenho entre um modelo de Regressão Linear e um modelo KNN.   

A estimativa baseia-se nas características físicas e de localização das propriedades.  Criar uma predição de preço confiável é de grande importância, 
pois ajuda compradores, vendedores, corretores e bancos (em processos de financiamento) a tomarem decisões embasadas, evitando a superavaliação ou 
subavaliação de um imóvel. 

--Dataset Utilizado

•	 Arquivo: kc_house_data.csv   
•	 Contexto: O conjunto de dados descreve o mercado imobiliário do condado de King County (onde fica a cidade de Seattle, EUA).   
•	 Tamanho: Aproximadamente 21 mil registros de vendas de imóveis ocorridas entre os anos de 2014 e 2015.   
•	 Dimensões (Shape): 21.613 linhas e 21 colunas.   
•	 Valores Nulos: O dataset é denso, apresentando apenas 2 valores nulos na coluna sqft_above.   

--Análise Exploratória de Dados (EDA) - Variável Alvo

•	A variável alvo da predição é o preço de venda (price).   
•	A distribuição dos preços apresenta uma forte assimetria à direita (skew de 4.02).   
•	Isso ocorre porque há poucas mansões com valores muito altos que puxam a média (US$ 540.088) para muito além da mediana (US$ 450.000).   
•	Esse comportamento é esperado no mercado imobiliário, mas serve de alerta indicando que a métrica RMSE será sensível a esses valores extremos 
no momento de avaliar os modelos.   

--Organização do Código e Estrutura do Projeto

Para manter o Jupyter Notebook focado exclusivamente na análise e no porquê de cada tomada de decisão, o código foi organizado de forma modular. As funções repetitivas foram extraídas para módulos Python reaproveitáveis dentro da pasta src/. 

A estrutura de scripts de apoio é a seguinte:

•	src/dataset.py: Lida com o carregamento dos dados brutos.   
•	src/features.py: Responsável pela limpeza dos dados, tratamento de outliers e engenharia de atributos (feature engineering).   
•	src/plots.py: Agrupa as funções dedicadas à visualização de dados (gráficos).   
•	src/modeling/train.py: Concentra a lógica de treino, avaliação e salvamento do modelo preditivo.   
•	src/config.py: Armazena os parâmetros centrais e os caminhos (paths) de diretórios do projeto.   


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

 Testar algoritmos não lineares mais robustos (Random Forest, Gradient Boosting/XGBoost, LightGBM) para capturar melhor as relações
 não lineares que hoje nem a Regressão Linear nem a KNN modelam muito bem.

 Tratar o segmento de imóveis de luxo (onde se concentra a maior dispersão de resíduos) separadamente, talvez com um modelo
 específico ou uma transformação logarítmica de preços.
