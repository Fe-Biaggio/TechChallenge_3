"""
Métricas de avaliação do modelo — classificação binária (alfabetizado
True/False), validação cruzada e importância de variáveis.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    """Métricas padrão para classificação binária, num dict serializável em JSON."""
    return {
        "acuracia": round(float(accuracy_score(y_true, y_pred)), 4),
        "precisao": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }


def validacao_cruzada(pipeline, X, y, n_splits: int = 5) -> dict:
    """
    StratifiedKFold no conjunto de treino — garante replicabilidade: se o
    desempenho variar muito entre os folds, o modelo não generaliza bem
    (overfit a uma partição específica dos dados), não é só sorte do split.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "roc_auc_por_fold": [round(float(s), 4) for s in scores],
        "roc_auc_medio": round(float(scores.mean()), 4),
        "roc_auc_desvio_padrao": round(float(scores.std()), 4),
    }


def feature_importance_random_forest(pipeline, top_n: int = 20) -> pd.DataFrame:
    """Importância nativa (MDI) do RandomForestClassifier, com nomes de coluna pós-encoding."""
    nomes = pipeline.named_steps["preprocessador"].get_feature_names_out()
    importancias = pipeline.named_steps["classificador"].feature_importances_
    df = pd.DataFrame({"feature": nomes, "importancia": importancias})
    return df.sort_values("importancia", ascending=False).head(top_n).reset_index(drop=True)


def coeficientes_regressao_logistica(pipeline, top_n: int = 20) -> pd.DataFrame:
    """Coeficientes (em escala padronizada, já que os numéricos passam por StandardScaler)."""
    nomes = pipeline.named_steps["preprocessador"].get_feature_names_out()
    coefs = pipeline.named_steps["classificador"].coef_[0]
    df = pd.DataFrame({"feature": nomes, "coeficiente": coefs})
    df["abs_coeficiente"] = df["coeficiente"].abs()
    return df.sort_values("abs_coeficiente", ascending=False).head(top_n).drop(columns="abs_coeficiente").reset_index(drop=True)
