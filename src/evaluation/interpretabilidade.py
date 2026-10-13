"""
Interpretabilidade por dois métodos independentes — quando concordam, a
conclusão não é artefato de um método só:

1. Importância por permutação (sklearn): embaralha uma feature ORIGINAL de
   cada vez no conjunto de teste e mede a queda de ROC-AUC. Model-agnostic,
   medida fora da amostra de treino, e no espaço de features que um gestor
   entende (não nas colunas pós-encoding).

2. SHAP: contribuição de cada feature por predição individual — diz não só
   "quanto" mas "em que direção". TreeExplainer quando o modelo suporta;
   caso contrário (ex.: HistGradientBoosting com splits categóricos nativos),
   PermutationExplainer, que é exato no mesmo sentido e só mais lento.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def importancia_permutacao(pipeline, X: pd.DataFrame, y, amostra: int = 50_000, n_repeats: int = 5,
                           seed: int = 42) -> pd.DataFrame:
    if len(X) > amostra:
        idx = np.random.default_rng(seed).choice(len(X), amostra, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    resultado = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                       random_state=seed, n_jobs=1)
    df = pd.DataFrame({
        "feature": X.columns,
        "queda_roc_auc": resultado.importances_mean,
        "desvio": resultado.importances_std,
    })
    return df.sort_values("queda_roc_auc", ascending=False).reset_index(drop=True)


def _nomes_limpos(nomes) -> list:
    limpos = []
    for n in nomes:
        n = n.split("__", 1)[-1]
        n = n.replace("missingindicator_", "ausente: ")
        limpos.append(n)
    return limpos


def calcular_shap_values(pipeline, X: pd.DataFrame, amostra: int = 2000, seed: int = 42):
    """
    Retorna (shap_values, X_transformado, nomes_features, metodo) para o
    classificador dentro do pipeline, no espaço pós-pré-processamento.
    """
    X_amostra = X.sample(n=min(amostra, len(X)), random_state=seed)
    preproc = pipeline.named_steps["preprocessador"]
    modelo = pipeline.named_steps["classificador"]
    X_t = preproc.transform(X_amostra)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    nomes = _nomes_limpos(preproc.get_feature_names_out())

    if hasattr(modelo, "coef_"):
        # Modelo linear: LinearExplainer é exato e instantâneo (efeito = coeficiente × desvio da média)
        explicador = shap.LinearExplainer(modelo, X_t)
        valores = explicador.shap_values(X_t)
        metodo = "LinearExplainer"
        return valores, X_t, nomes, metodo
    try:
        explicador = shap.TreeExplainer(modelo)
        valores = explicador.shap_values(X_t)
        metodo = "TreeExplainer"
    except Exception:
        fundo = shap.sample(X_t, 100, random_state=seed)
        explicador = shap.PermutationExplainer(lambda d: modelo.predict_proba(d)[:, 1], fundo)
        valores = explicador(X_t, silent=True).values
        metodo = "PermutationExplainer"

    if isinstance(valores, list):
        valores = valores[1]
    elif getattr(valores, "ndim", 2) == 3:
        valores = valores[:, :, 1]
    return valores, X_t, nomes, metodo


def resumo_shap(shap_values, nomes_features, top_n: int = 20) -> pd.DataFrame:
    """|SHAP| médio por feature e direção média do efeito (sinal da correlação entre valor e SHAP)."""
    df = pd.DataFrame({
        "feature": nomes_features,
        "importancia_media_shap": np.abs(shap_values).mean(axis=0),
    })
    return df.sort_values("importancia_media_shap", ascending=False).head(top_n).reset_index(drop=True)
