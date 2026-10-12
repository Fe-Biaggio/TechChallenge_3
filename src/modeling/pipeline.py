"""
Pipeline scikit-learn de pré-processamento + modelo — um único objeto
`Pipeline` que recebe o DataFrame bruto de features e produz a predição,
sem nenhum passo de transformação feito "na mão" fora dele (ver PDF:
"Integração do pré-processamento diretamente ao modelo").

Dois modelos são comparados (ver src/modeling/treinar.py):
  - Regressão Logística: baseline linear, interpretável por coeficiente
  - Random Forest: não-linear, captura interação entre variáveis,
    feature_importances_ nativo
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.dataset import TODAS_CATEGORICAS, TODAS_NUMERICAS


def construir_preprocessador() -> ColumnTransformer:
    """
    Imputação + transformação, isoladas em um ColumnTransformer:
      - Numéricas: mediana (robusta a outlier) + padronização
      - Categóricas: constante "Desconhecido" (ausência de contexto
        municipal é informação válida, não "não sei o que pôr aqui") +
        one-hot (ignora categoria nunca vista no treino, ex. rede=Federal)
    """
    transformador_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    transformador_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", transformador_numerico, TODAS_NUMERICAS),
        ("cat", transformador_categorico, TODAS_CATEGORICAS),
    ])


def construir_pipeline_logistica() -> Pipeline:
    return Pipeline([
        ("preprocessador", construir_preprocessador()),
        ("classificador", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def construir_pipeline_random_forest() -> Pipeline:
    return Pipeline([
        ("preprocessador", construir_preprocessador()),
        ("classificador", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


MODELOS_CANDIDATOS = {
    "regressao_logistica": construir_pipeline_logistica,
    "random_forest": construir_pipeline_random_forest,
}
