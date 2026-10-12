"""
Interpretabilidade via SHAP — complementa feature_importances_ (que só diz
"quanto", não "em que direção"). Calculado sobre uma amostra, não o dataset
inteiro: SHAP em TreeExplainer é rápido, mas não há necessidade de rodar em
~1,8M linhas para um gráfico — uma amostra estratificada já estabiliza a
estimativa.
"""
import numpy as np
import pandas as pd
import shap


def calcular_shap_values(pipeline, X: pd.DataFrame, amostra: int = 5000, seed: int = 42):
    """
    Retorna (shap_values, X_transformado_amostra, nomes_features) para o
    classificador de árvore dentro do pipeline (TreeExplainer).
    """
    X_amostra = X.sample(n=min(amostra, len(X)), random_state=seed)
    X_transformado = pipeline.named_steps["preprocessador"].transform(X_amostra)
    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()

    nomes = pipeline.named_steps["preprocessador"].get_feature_names_out()
    explicador = shap.TreeExplainer(pipeline.named_steps["classificador"])
    shap_values = explicador.shap_values(X_transformado)

    # Para classificador binário, shap_values pode vir como lista [classe0, classe1] ou array 3D
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_transformado, nomes


def resumo_shap(shap_values, nomes_features, top_n: int = 15) -> pd.DataFrame:
    """Importância média |SHAP| por feature — mesmo princípio de feature_importances_, mas model-agnostic."""
    importancia_media = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": nomes_features, "importancia_media_shap": importancia_media})
    return df.sort_values("importancia_media_shap", ascending=False).head(top_n).reset_index(drop=True)
