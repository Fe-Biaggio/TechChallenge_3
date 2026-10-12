"""
Orquestra o treino: monta o dataset, separa treino/validação/teste,
treina os modelos candidatos, valida com cross-validation, escolhe o
melhor por ROC-AUC de validação, reavalia no teste (hold-out nunca visto),
e salva modelo + métricas + gráficos.

Uso:
    python -m src.modeling.treinar
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metricas import (  # noqa: E402
    calcular_metricas,
    coeficientes_regressao_logistica,
    feature_importance_random_forest,
    validacao_cruzada,
)
from src.modeling.pipeline import MODELOS_CANDIDATOS  # noqa: E402
from src.preprocessing.dataset import COLUNA_ALVO, TODAS_FEATURES, montar_dataset_modelagem  # noqa: E402
from src.visualization import graficos  # noqa: E402

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42


def _t(inicio: float) -> str:
    return f"{time.time() - inicio:.1f}s"


def main():
    t0 = time.time()
    print("=" * 60)
    print("TREINO DO MODELO — Predição de Alfabetização (Fase 3)")
    print("=" * 60)

    df = montar_dataset_modelagem()
    X = df[TODAS_FEATURES]
    y = df[COLUNA_ALVO].astype(int)

    # 70% treino / 15% validação / 15% teste, estratificado pelo alvo
    X_treino, X_resto, y_treino, y_resto = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_resto, y_resto, test_size=0.50, stratify=y_resto, random_state=SEED
    )
    print(f"\n[split] treino={len(X_treino):,} | validação={len(X_val):,} | teste={len(X_teste):,}")
    print(f"[split] taxa alfabetizados — treino={y_treino.mean():.3f} val={y_val.mean():.3f} teste={y_teste.mean():.3f}")

    resultados_validacao = {}
    pipelines_treinados = {}

    for nome, construtor in MODELOS_CANDIDATOS.items():
        t1 = time.time()
        print(f"\n--- {nome} ---")
        pipe = construtor()
        pipe.fit(X_treino, y_treino)
        print(f"[{nome}] treinado em {_t(t1)}")
        pipelines_treinados[nome] = pipe

        y_pred_val = pipe.predict(X_val)
        y_proba_val = pipe.predict_proba(X_val)[:, 1]
        metricas_val = calcular_metricas(y_val, y_pred_val, y_proba_val)
        print(f"[{nome}] validação: {metricas_val}")

        t_cv = time.time()
        cv = validacao_cruzada(pipe, X_treino, y_treino, n_splits=5)
        print(f"[{nome}] validação cruzada (5-fold, {_t(t_cv)}): ROC-AUC = {cv['roc_auc_medio']} ± {cv['roc_auc_desvio_padrao']}")

        resultados_validacao[nome] = {"metricas_validacao": metricas_val, "validacao_cruzada": cv}

    melhor_modelo = max(resultados_validacao, key=lambda k: resultados_validacao[k]["metricas_validacao"]["roc_auc"])
    print(f"\n>>> Melhor modelo (ROC-AUC validação): {melhor_modelo}")

    pipe_final = pipelines_treinados[melhor_modelo]
    y_pred_teste = pipe_final.predict(X_teste)
    y_proba_teste = pipe_final.predict_proba(X_teste)[:, 1]
    metricas_teste = calcular_metricas(y_teste, y_pred_teste, y_proba_teste)
    print(f"\n[{melhor_modelo}] TESTE (hold-out nunca visto): {metricas_teste}")

    # ─── Gráficos ───
    print("\n[gráficos] gerando...")
    graficos.grafico_distribuicao_alvo(df)
    graficos.grafico_taxa_por_regiao(df)
    from src.preprocessing.dataset import TODAS_NUMERICAS
    graficos.grafico_correlacao_numericas(df, TODAS_NUMERICAS)
    graficos.grafico_matriz_confusao(y_teste, y_pred_teste, f"Matriz de Confusão — {melhor_modelo} (teste)")
    graficos.grafico_curva_roc(y_teste, y_proba_teste, f"Curva ROC — {melhor_modelo} (teste)")

    if melhor_modelo == "random_forest":
        imp = feature_importance_random_forest(pipe_final)
        graficos.grafico_feature_importance(imp, "Importância de features — Random Forest (MDI)", "06_feature_importance.png")
        imp.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

        from src.evaluation.interpretabilidade import calcular_shap_values, resumo_shap
        t_shap = time.time()
        shap_values, X_transf, nomes = calcular_shap_values(pipe_final, X_teste, amostra=5000)
        graficos.grafico_shap_summary(shap_values, X_transf, nomes)
        resumo_shap(shap_values, nomes).to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
        print(f"[shap] calculado em {_t(t_shap)}")
    else:
        coefs = coeficientes_regressao_logistica(pipe_final)
        graficos.grafico_feature_importance(
            coefs.rename(columns={"coeficiente": "importancia"})[["feature", "coeficiente"]],
            "Coeficientes — Regressão Logística (padronizados)", "06_feature_importance.png",
        )
        coefs.to_csv(REPORTS_DIR / "coeficientes_logistica.csv", index=False)

    # ─── Persistência ───
    caminho_modelo = MODELS_DIR / f"modelo_{melhor_modelo}.joblib"
    joblib.dump(pipe_final, caminho_modelo)
    print(f"\n[salvo] modelo -> {caminho_modelo}")

    relatorio = {
        "modelo_escolhido": melhor_modelo,
        "criterio_escolha": "maior ROC-AUC no conjunto de validação",
        "resultados_validacao_por_modelo": resultados_validacao,
        "metricas_teste_holdout": metricas_teste,
        "n_treino": len(X_treino),
        "n_validacao": len(X_val),
        "n_teste": len(X_teste),
        "features": TODAS_FEATURES,
        "tempo_total_segundos": round(time.time() - t0, 1),
    }
    caminho_relatorio = REPORTS_DIR / "metricas_modelo.json"
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"[salvo] métricas -> {caminho_relatorio}")

    print(f"\nConcluído em {_t(t0)}")
    return relatorio


if __name__ == "__main__":
    main()
