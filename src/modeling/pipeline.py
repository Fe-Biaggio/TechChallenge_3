"""
Pipelines scikit-learn: pré-processamento + modelo num único objeto, sem
nenhuma transformação feita "na mão" fora dele — então o que é aprendido no
treino (medianas, categorias, escalas) nunca vê dado de validação/teste.

Dois pré-processadores, um por família de modelo:

  Linear (Regressão Logística)
    - categóricas: imputação constante "Desconhecido" + one-hot
    - contagens: imputação mediana + log1p (cauda longa) + padronização
    - numéricas: imputação mediana COM indicador de ausência + padronização

  Árvores (Random Forest, HistGradientBoosting)
    - categóricas: imputação constante + codificação ordinal (o HGB as trata
      como categóricas nativas — colunas vêm primeiro no ColumnTransformer
      justamente para os índices serem fixos)
    - contagens: imputação mediana + log1p
    - numéricas: imputação mediana COM indicador de ausência (árvores não
      precisam de escala)

O indicador de ausência importa aqui: ~23% dos alunos estão em municípios
sem microdados no ano anterior — "não havia dado" é informação, e a
mediana sozinha apagaria isso.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocessing.dataset import FEATURES_CATEGORICAS, FEATURES_CONTAGEM, FEATURES_NUMERICAS

SEED = 42


def _log1p():
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def preprocessador_linear() -> ColumnTransformer:
    """ColumnTransformer para modelos lineares: imputação, one-hot, log1p e padronização."""
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imputador", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES_CATEGORICAS),
        ("cont", Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("log", _log1p()),
            ("escala", StandardScaler()),
        ]), FEATURES_CONTAGEM),
        ("num", Pipeline([
            ("imputador", SimpleImputer(strategy="median", add_indicator=True)),
            ("escala", StandardScaler()),
        ]), FEATURES_NUMERICAS),
    ])


def preprocessador_arvore() -> ColumnTransformer:
    """ColumnTransformer para árvores: imputação, codificação ordinal e log1p, sem escala."""
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imputador", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), FEATURES_CATEGORICAS),
        ("cont", Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("log", _log1p()),
        ]), FEATURES_CONTAGEM),
        ("num", SimpleImputer(strategy="median", add_indicator=True), FEATURES_NUMERICAS),
    ])


# Categóricas são as primeiras colunas de saída do preprocessador_arvore
INDICES_CATEGORICOS_ARVORE = list(range(len(FEATURES_CATEGORICAS)))


def pipeline_baseline() -> Pipeline:
    """Referência trivial: sempre a classe majoritária / probabilidade = prevalência."""
    return Pipeline([
        ("preprocessador", preprocessador_arvore()),
        ("classificador", DummyClassifier(strategy="prior")),
    ])


def pipeline_logistica(**params) -> Pipeline:
    """
    Regularização L2; a força (C) é escolhida por busca em validação cruzada
    agrupada (ESPACO_BUSCA_LOGISTICA). C menor = coeficientes mais encolhidos.
    """
    base = dict(C=1.0, max_iter=2000, random_state=SEED)
    base.update(params)
    return Pipeline([
        ("preprocessador", preprocessador_linear()),
        ("classificador", LogisticRegression(**base)),
    ])


def pipeline_random_forest() -> Pipeline:
    """Random Forest regularizado (profundidade e folha mínima limitadas) com o pré-processador de árvores."""
    return Pipeline([
        ("preprocessador", preprocessador_arvore()),
        ("classificador", RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=100,
            max_features="sqrt", n_jobs=-1, random_state=SEED,
        )),
    ])


def pipeline_hist_gradient_boosting(**params) -> Pipeline:
    """
    Early stopping é a principal defesa contra overfitting aqui: o treino para
    quando a perda numa fração de validação interna para de melhorar.
    """
    base = dict(
        categorical_features=INDICES_CATEGORICOS_ARVORE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=600,
        random_state=SEED,
    )
    base.update(params)
    return Pipeline([
        ("preprocessador", preprocessador_arvore()),
        ("classificador", HistGradientBoostingClassifier(**base)),
    ])


# Espaço de busca do HGB (RandomizedSearchCV — ver src/modeling/treinar.py)
ESPACO_BUSCA_HGB = {
    "classificador__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "classificador__max_leaf_nodes": [15, 31, 63, 127],
    "classificador__min_samples_leaf": [20, 100, 500, 2000],
    "classificador__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "classificador__max_depth": [None, 6, 10],
}

# Busca da Regressão Logística (GridSearchCV — ver src/modeling/treinar.py)
ESPACO_BUSCA_LOGISTICA = {
    "classificador__C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

CANDIDATOS = {
    "baseline_prevalencia": pipeline_baseline,
    "regressao_logistica": pipeline_logistica,
    "random_forest": pipeline_random_forest,
    "hist_gradient_boosting": pipeline_hist_gradient_boosting,
}
