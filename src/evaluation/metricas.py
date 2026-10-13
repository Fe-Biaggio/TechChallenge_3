"""
Métricas de avaliação. A classe de interesse para política pública é
"não alfabetizado" (risco) — por isso, além das métricas padrão sobre a
classe positiva, reportamos precisão/recall da classe de risco explicitamente.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    """Métricas de classificação binária, serializáveis em JSON."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    risco_true, risco_pred = 1 - y_true, 1 - y_pred
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "acuracia": round(float(accuracy_score(y_true, y_pred)), 4),
        "precisao": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precisao_risco": round(float(precision_score(risco_true, risco_pred, zero_division=0)), 4),
        "recall_risco": round(float(recall_score(risco_true, risco_pred, zero_division=0)), 4),
        "f1_risco": round(float(f1_score(risco_true, risco_pred, zero_division=0)), 4),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }


# Mantido por compatibilidade com o nome usado antes desta revisão
def validacao_cruzada(pipeline, X, y, n_splits: int = 5, groups=None) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", groups=groups)
    return {
        "roc_auc_por_fold": [round(float(s), 4) for s in scores],
        "roc_auc_medio": round(float(scores.mean()), 4),
        "roc_auc_desvio_padrao": round(float(scores.std()), 4),
    }


def analise_limiar(y_true, y_proba) -> pd.DataFrame:
    """
    Precisão e recall da classe de RISCO (não alfabetizado) para cada limiar
    de P(risco). Orienta o uso operacional: um programa de reforço com vagas
    limitadas quer precisão alta; uma triagem ampla quer recall alto.
    """
    risco_true = 1 - np.asarray(y_true).astype(int)
    p_risco = 1 - np.asarray(y_proba)
    precisao, recall, limiares = precision_recall_curve(risco_true, p_risco)
    df = pd.DataFrame({"limiar_p_risco": limiares, "precisao_risco": precisao[:-1], "recall_risco": recall[:-1]})
    df["f1_risco"] = 2 * df.precisao_risco * df.recall_risco / (df.precisao_risco + df.recall_risco).replace(0, np.nan)
    # Amostra regular de 50 pontos para tabela/gráfico legível
    idx = np.linspace(0, len(df) - 1, 50).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def analise_priorizacao(y_true, y_proba, fracoes=(0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """
    Uso operacional realista: um programa de reforço com capacidade para
    atender X% dos alunos atende os X% de maior risco previsto. Para cada X,
    reporta a taxa real de "não alfabetizado" dentro do grupo priorizado, a
    fração de todos os alunos em risco que ele cobre (recall) e o lift sobre
    a taxa média (quantas vezes melhor que escolher alunos ao acaso).
    """
    risco_true = 1 - np.asarray(y_true).astype(int)
    p_risco = 1 - np.asarray(y_proba)
    ordem = np.argsort(-p_risco)
    taxa_base = risco_true.mean()
    linhas = []
    for f in fracoes:
        k = max(1, int(round(len(ordem) * f)))
        selecionados = ordem[:k]
        taxa_grupo = risco_true[selecionados].mean()
        linhas.append({
            "fracao_priorizada": f,
            "limiar_p_risco": round(float(p_risco[ordem[k - 1]]), 4),
            "taxa_risco_no_grupo": round(float(taxa_grupo), 4),
            "recall_risco": round(float(risco_true[selecionados].sum() / risco_true.sum()), 4),
            "lift": round(float(taxa_grupo / taxa_base), 3),
        })
    return pd.DataFrame(linhas)


def desempenho_por_subgrupo(df: pd.DataFrame, coluna_grupo: str, coluna_alvo: str, coluna_proba: str,
                            min_n: int = 1000) -> pd.DataFrame:
    """ROC-AUC e calibração por subgrupo — onde o modelo funciona pior, e se superestima/subestima."""
    linhas = []
    for grupo, sub in df.groupby(coluna_grupo, observed=True):
        if len(sub) < min_n or sub[coluna_alvo].nunique() < 2:
            continue
        linhas.append({
            coluna_grupo: grupo,
            "n_alunos": len(sub),
            "taxa_real": round(float(sub[coluna_alvo].mean() * 100), 1),
            "taxa_prevista": round(float(sub[coluna_proba].mean() * 100), 1),
            "roc_auc": round(float(roc_auc_score(sub[coluna_alvo], sub[coluna_proba])), 4),
        })
    return pd.DataFrame(linhas).sort_values("roc_auc").reset_index(drop=True)


def feature_importance_random_forest(pipeline, top_n: int = 20) -> pd.DataFrame:
    nomes = pipeline.named_steps["preprocessador"].get_feature_names_out()
    importancias = pipeline.named_steps["classificador"].feature_importances_
    df = pd.DataFrame({"feature": nomes, "importancia": importancias})
    return df.sort_values("importancia", ascending=False).head(top_n).reset_index(drop=True)


def coeficientes_regressao_logistica(pipeline, top_n: int = 20) -> pd.DataFrame:
    nomes = pipeline.named_steps["preprocessador"].get_feature_names_out()
    coefs = pipeline.named_steps["classificador"].coef_[0]
    df = pd.DataFrame({"feature": nomes, "coeficiente": coefs})
    df["abs_coeficiente"] = df["coeficiente"].abs()
    return (df.sort_values("abs_coeficiente", ascending=False).head(top_n)
            .drop(columns="abs_coeficiente").reset_index(drop=True))
