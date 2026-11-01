"""
Interpretabilidade por dois métodos independentes — quando concordam, a
conclusão não é artefato de um método só:

1. Importância por permutação (sklearn): embaralha uma feature ORIGINAL de
   cada vez no conjunto de teste e mede a queda de ROC-AUC. Model-agnostic,
   medida fora da amostra de treino, e no espaço de features que um gestor
   entende (não nas colunas pós-encoding).

2. SHAP: contribuição de cada feature por predição individual — diz não só
   "quanto" mas "em que direção". LinearExplainer para modelos lineares,
   TreeExplainer quando o modelo de árvore suporta; caso contrário,
   PermutationExplainer, que é exato no mesmo sentido e só mais lento.

3. Ablação por grupo: retreina o modelo sem cada GRUPO de features (e só com
   ele) em validação cruzada agrupada. Responde "quais fatores mais impactam"
   de forma robusta a correlação entre features do mesmo grupo — a
   permutação de uma feature isolada subestima o grupo quando as outras
   do grupo carregam a mesma informação.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def importancia_permutacao(pipeline, X: pd.DataFrame, y, amostra: int = 50_000, n_repeats: int = 5,
                           seed: int = 42) -> pd.DataFrame:
    """Queda média de ROC-AUC ao embaralhar cada feature original (amostra do conjunto avaliado)."""
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


# Grupos de features por tipo de fator — base da ablação
GRUPOS_FEATURES = {
    "Histórico educacional do município (ano anterior)": [
        "taxa_alfabetizacao_ant", "nivel_alfabetizacao_ant", "participacao_ant",
        "prof_media_ant", "prof_desvio_ant", "prof_p25_ant",
    ],
    "Meta e esforço para atingi-la": ["meta_ano", "esforco_meta"],
    "IDHM (socioeconômico)": ["idhm_educacao", "idhm_renda", "idhm_longevidade"],
    "Escola e rede": ["rede_label", "escola_n_alunos"],
    "Território e porte do município": ["capital_uf", "amazonia_legal", "mun_n_alunos_ant"],
}


def neutralizar(X: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """
    Remove a informação de `colunas` sem mudar o formato de X: numéricas viram
    constante 0 e categóricas viram uma categoria única. Uma coluna constante
    não tem variância nem ausentes — o pipeline a mantém, mas o modelo não
    consegue extrair nada dela. Equivale a treinar sem as colunas, com o mesmo
    pré-processador.
    """
    X = X.copy()
    for c in colunas:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = 0.0
        else:
            X[c] = "Desconhecido"
    return X


def ablacao_por_grupo(construtor, X: pd.DataFrame, y, grupos, cv, grupos_features: dict = None) -> pd.DataFrame:
    """
    ROC-AUC de validação cruzada (agrupada) com todas as features, sem cada
    grupo e só com cada grupo. `construtor()` devolve um pipeline novo.
    """
    from sklearn.model_selection import cross_val_score

    grupos_features = grupos_features or GRUPOS_FEATURES
    todas = [c for cols in grupos_features.values() for c in cols]

    def _auc(X_):
        return float(cross_val_score(construtor(), X_, y, groups=grupos, cv=cv, scoring="roc_auc").mean())

    completo = _auc(X)
    linhas = []
    for nome, cols in grupos_features.items():
        sem = _auc(neutralizar(X, cols))
        so = _auc(neutralizar(X, [c for c in todas if c not in cols]))
        linhas.append({
            "grupo": nome,
            "n_features": len(cols),
            "roc_auc_sem_o_grupo": round(sem, 4),
            "perda_ao_remover": round(completo - sem, 4),
            "roc_auc_so_o_grupo": round(so, 4),
        })
    df = pd.DataFrame(linhas).sort_values("perda_ao_remover", ascending=False).reset_index(drop=True)
    df.attrs["roc_auc_completo"] = round(completo, 4)
    return df
