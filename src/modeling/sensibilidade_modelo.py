"""
Análise de sensibilidade da escolha de modelo (D21 em docs/decisoes_analiticas.md).

A comparação pareada na validação cruzada mostra o HistGradientBoosting
melhor que a Regressão Logística em todos os folds, por ~0,004 de ROC-AUC —
estatisticamente detectável, mas pequeno. A pergunta que importa para o uso
é outra: trocar de modelo mudaria alguma DECISÃO? Este script treina o HGB
(com os hiperparâmetros da busca) no mesmo treino e compara os dois no mesmo
teste, nas métricas de uso: ROC-AUC, calibração, lift da priorização,
agregado municipal e sobreposição dos municípios priorizados.

Não altera a seleção (feita antes, só com validação cruzada); é uma
checagem posterior, para documentar o custo de ter escolhido o mais simples.

Uso:
    python -m src.modeling.sensibilidade_modelo     # ~5 min
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation import aplicacao_estrategica as estrategia  # noqa: E402
from src.evaluation.metricas import analise_priorizacao  # noqa: E402
from src.modeling.pipeline import pipeline_hist_gradient_boosting, pipeline_logistica  # noqa: E402
from src.modeling.treinar import _cv_agrupado  # noqa: E402
from src.preprocessing.dataset import COLUNA_ALVO, COLUNA_GRUPO, TODAS_FEATURES, montar_dataset_modelagem  # noqa: E402

REPORTS = ROOT / "reports"


def _resumo(nome, y_te, proba, df_te):
    prior = analise_priorizacao(y_te, proba).set_index("fracao_priorizada")
    mun = estrategia.agregar_por_municipio(
        df_te.assign(proba=proba)[df_te["rede_label"] == "Municipal"], "proba", COLUNA_ALVO)
    mun = mun[mun["n_alunos"] >= estrategia.MIN_ALUNOS_MUNICIPIO]
    return {
        "modelo": nome,
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "lift_top20": float(prior.loc[0.20, "lift"]),
        "correlacao_municipal": round(float(mun[["taxa_prevista", "taxa_real"]].corr().iloc[0, 1]), 4),
        "mae_municipal_pp": round(float((mun.taxa_prevista - mun.taxa_real).abs().mean()), 2),
    }, mun.set_index("id_municipio")["taxa_prevista"]


def main():
    """Treina os dois modelos no mesmo treino, compara no mesmo teste e grava o JSON."""
    metricas = json.loads((REPORTS / "metricas_modelo.json").read_text(encoding="utf-8"))
    df = montar_dataset_modelagem(verbose=False)
    X, y, grupos = df[TODAS_FEATURES], df[COLUNA_ALVO].astype(int), df[COLUNA_GRUPO]
    idx_tr, idx_te = next(_cv_agrupado(5).split(X, y, grupos))  # mesmo split do treino
    X_tr, y_tr, X_te, y_te = X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_te]
    df_te = df.iloc[idx_te]

    lr = pipeline_logistica(**metricas["hiperparametros_logistica"]).fit(X_tr, y_tr)
    hgb = pipeline_hist_gradient_boosting(**metricas["hiperparametros_hgb"]).fit(X_tr, y_tr)

    r_lr, mun_lr = _resumo("regressao_logistica", y_te, lr.predict_proba(X_te)[:, 1], df_te)
    r_hgb, mun_hgb = _resumo("hist_gradient_boosting", y_te, hgb.predict_proba(X_te)[:, 1], df_te)

    # Os dois modelos priorizariam os mesmos municípios? (20% de maior risco previsto)
    comuns = mun_lr.index.intersection(mun_hgb.index)
    n_top = int(round(len(comuns) * 0.20))
    top_lr = set(mun_lr.loc[comuns].nsmallest(n_top).index)
    top_hgb = set(mun_hgb.loc[comuns].nsmallest(n_top).index)
    resultado = {
        "comparacao_no_teste": [r_lr, r_hgb],
        "sobreposicao_top20_municipios": round(len(top_lr & top_hgb) / n_top, 4),
        "correlacao_taxas_municipais_entre_modelos": round(float(np.corrcoef(mun_lr.loc[comuns],
                                                                             mun_hgb.loc[comuns])[0, 1]), 4),
        "municipios_comparados": int(len(comuns)),
    }
    (REPORTS / "sensibilidade_modelo.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
