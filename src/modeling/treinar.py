"""
Orquestra treino, avaliação e aplicação estratégica do modelo da Fase 3.

Desenho da validação (ver docs/decisoes_analiticas.md):

  1. Hold-out de TESTE por município: 20% dos municípios ficam de fora de
     tudo — busca de hiperparâmetros, comparação de modelos e treino. Testar
     em municípios nunca vistos mede generalização territorial de verdade;
     um split aleatório por aluno colocaria colegas do mesmo município no
     treino e no teste e superestimaria o desempenho.
  2. Busca de hiperparâmetros (HistGradientBoosting) e comparação entre
     candidatos — incluindo um baseline trivial — por validação cruzada
     estratificada e AGRUPADA por município, numa subamostra do treino.
  3. O vencedor é retreinado no treino completo e avaliado UMA vez no teste.
  4. Predições out-of-fold (cada aluno previsto por um modelo que nunca viu
     o município dele) alimentam o ranking de risco e o backtest de metas.
  5. Modelo final, treinado com todos os alunos de 2024, faz a projeção
     para a meta de 2025.

Uso:
    python -m src.modeling.treinar                  # tudo (~30-40 min)
    python -m src.modeling.treinar --so-estrategia  # só refaz a aplicação estratégica
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# Console do Windows usa cp1252 quando a saída é redirecionada — sem isso, "≥" derruba o log
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation import aplicacao_estrategica as estrategia  # noqa: E402
from src.evaluation.interpretabilidade import (  # noqa: E402
    ablacao_por_grupo,
    calcular_shap_values,
    importancia_permutacao,
    resumo_shap,
)
from src.evaluation.metricas import (  # noqa: E402
    analise_limiar,
    analise_priorizacao,
    bootstrap_por_grupo,
    calcular_metricas,
    coeficientes_regressao_logistica,
    comparacao_pareada,
    desempenho_por_subgrupo,
)
from src.modeling.pipeline import (  # noqa: E402
    CANDIDATOS,
    ESPACO_BUSCA_HGB,
    ESPACO_BUSCA_LOGISTICA,
    SEED,
    pipeline_hist_gradient_boosting,
    pipeline_logistica,
)
from src.preprocessing.dataset import (  # noqa: E402
    ANO_TREINO,
    COLUNA_ALVO,
    COLUNA_GRUPO,
    FEATURES_CONTAGEM,
    FEATURES_NUMERICAS,
    TODAS_FEATURES,
    montar_dataset_modelagem,
    montar_dataset_projecao,
)
from src.visualization import graficos  # noqa: E402

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PREDICOES_DIR = ROOT / "data" / "modelagem"
for _d in (MODELS_DIR, REPORTS_DIR, PREDICOES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ANO_PROJECAO = ANO_TREINO + 1
TAMANHO_SUBAMOSTRA = 300_000
N_ITER_BUSCA = 15


def log(msg: str) -> None:
    """Print com flush imediato (o log acompanha o progresso de execuções longas)."""
    print(msg, flush=True)


def _cv_agrupado(n_splits: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _construir(nome: str, params: dict):
    """Pipeline novo do candidato `nome`, com os hiperparâmetros escolhidos na busca (se houver)."""
    return CANDIDATOS[nome](**params.get(nome, {}))


# Do mais simples para o mais complexo — usado pela regra de 1 desvio padrão
ORDEM_COMPLEXIDADE = ["regressao_logistica", "random_forest", "hist_gradient_boosting"]


def escolher_modelo(comparacao: pd.DataFrame) -> tuple:
    """
    Regra de 1 desvio padrão: entre os candidatos cuja ROC-AUC média de
    validação está a no máximo 1 desvio padrão do melhor, escolhe o MAIS
    SIMPLES. Diferenças menores que a variação entre folds não são evidência
    de que o modelo mais complexo generaliza melhor — e o mais simples tem,
    em geral, menor gap treino-validação e é mais fácil de explicar.
    """
    reais = comparacao[comparacao.modelo != "baseline_prevalencia"].set_index("modelo")
    lider = reais["cv_roc_auc_medio"].idxmax()
    corte = reais.loc[lider, "cv_roc_auc_medio"] - reais.loc[lider, "cv_roc_auc_desvio"]
    elegiveis = [m for m in ORDEM_COMPLEXIDADE if m in reais.index and reais.loc[m, "cv_roc_auc_medio"] >= corte]
    escolhido = elegiveis[0]
    if escolhido == lider:
        return escolhido, "maior ROC-AUC de validação, sem candidato mais simples dentro de 1 desvio padrão"
    return escolhido, (f"mais simples dentro de 1 desvio padrão do melhor ({lider}: "
                       f"{reais.loc[lider, 'cv_roc_auc_medio']:.4f} ± {reais.loc[lider, 'cv_roc_auc_desvio']:.4f}; "
                       f"{escolhido}: {reais.loc[escolhido, 'cv_roc_auc_medio']:.4f})")


def treinar_e_avaliar() -> dict:
    """Etapas 1 a 5 do desenho da validação (docstring do módulo); grava métricas, gráficos e modelo."""
    t0 = time.time()
    log("=" * 64)
    log("FASE 3 — Predição de alfabetização por aluno")
    log("=" * 64)

    df = montar_dataset_modelagem()
    X, y, grupos = df[TODAS_FEATURES], df[COLUNA_ALVO].astype(int), df[COLUNA_GRUPO]

    # 1. Hold-out por município
    idx_treino, idx_teste = next(_cv_agrupado(5).split(X, y, grupos))
    X_tr, y_tr, g_tr = X.iloc[idx_treino], y.iloc[idx_treino], grupos.iloc[idx_treino]
    X_te, y_te = X.iloc[idx_teste], y.iloc[idx_teste]
    sobreposicao = set(g_tr) & set(grupos.iloc[idx_teste])
    assert not sobreposicao, "vazamento: município presente no treino e no teste"
    log(f"\n[split] treino {len(X_tr):,} alunos / {g_tr.nunique():,} municípios | "
        f"teste {len(X_te):,} alunos / {grupos.iloc[idx_teste].nunique():,} municípios (nenhum em comum)")

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(X_tr), size=min(TAMANHO_SUBAMOSTRA, len(X_tr)), replace=False)
    X_sub, y_sub, g_sub = X_tr.iloc[sub], y_tr.iloc[sub], g_tr.iloc[sub]

    # 2a. Busca de hiperparâmetros do HGB
    t = time.time()
    busca = RandomizedSearchCV(
        pipeline_hist_gradient_boosting(), ESPACO_BUSCA_HGB, n_iter=N_ITER_BUSCA, scoring="roc_auc",
        cv=_cv_agrupado(3), random_state=SEED, n_jobs=1, refit=False,
    )
    busca.fit(X_sub, y_sub, groups=g_sub)
    params_hgb = {k.replace("classificador__", ""): v for k, v in busca.best_params_.items()}
    tuning = (pd.DataFrame(busca.cv_results_)
              .sort_values("rank_test_score")
              [["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]])
    tuning.to_csv(REPORTS_DIR / "tuning_hist_gradient_boosting.csv", index=False)
    log(f"\n[busca] {N_ITER_BUSCA} configurações × 3 folds em {time.time() - t:.0f}s — "
        f"melhor ROC-AUC {busca.best_score_:.4f}: {params_hgb}")

    # 2a'. Busca da regularização da Regressão Logística (mesmo desenho de CV)
    t = time.time()
    busca_lr = GridSearchCV(pipeline_logistica(), ESPACO_BUSCA_LOGISTICA, scoring="roc_auc",
                            cv=_cv_agrupado(3), n_jobs=1, refit=False)
    busca_lr.fit(X_sub, y_sub, groups=g_sub)
    params_lr = {k.replace("classificador__", ""): v for k, v in busca_lr.best_params_.items()}
    (pd.DataFrame(busca_lr.cv_results_).sort_values("rank_test_score")
     [["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]]
     .to_csv(REPORTS_DIR / "tuning_regressao_logistica.csv", index=False))
    log(f"[busca] regressão logística: {len(busca_lr.cv_results_['params'])} valores de C × 3 folds em "
        f"{time.time() - t:.0f}s — melhor ROC-AUC {busca_lr.best_score_:.4f}: {params_lr}")
    params = {"hist_gradient_boosting": params_hgb, "regressao_logistica": params_lr}

    # 2b. Comparação entre candidatos (mesma CV agrupada, mesma subamostra)
    linhas, scores_por_fold = [], {}
    for nome in CANDIDATOS:
        t = time.time()
        res = cross_validate(_construir(nome, params), X_sub, y_sub, groups=g_sub, cv=_cv_agrupado(5),
                             scoring="roc_auc", return_train_score=True, n_jobs=1)
        scores_por_fold[nome] = res["test_score"]
        linhas.append({
            "modelo": nome,
            "cv_roc_auc_medio": round(float(res["test_score"].mean()), 4),
            "cv_roc_auc_desvio": round(float(res["test_score"].std()), 4),
            "treino_roc_auc_medio": round(float(res["train_score"].mean()), 4),
            "gap_treino_validacao": round(float(res["train_score"].mean() - res["test_score"].mean()), 4),
            "tempo_fit_medio_s": round(float(res["fit_time"].mean()), 1),
        })
        log(f"[cv] {nome:24s} ROC-AUC {linhas[-1]['cv_roc_auc_medio']:.4f} ± {linhas[-1]['cv_roc_auc_desvio']:.4f} "
            f"| treino {linhas[-1]['treino_roc_auc_medio']:.4f} | {time.time() - t:.0f}s")
    comparacao = pd.DataFrame(linhas)
    comparacao.to_csv(REPORTS_DIR / "comparacao_modelos.csv", index=False)

    melhor, justificativa = escolher_modelo(comparacao)
    log(f"\n>>> Modelo escolhido: {melhor} — {justificativa}")

    # Comparação pareada (mesmos folds): cada outro candidato real menos o escolhido
    pd.DataFrame(scores_por_fold).rename_axis("fold").to_csv(REPORTS_DIR / "comparacao_modelos_por_fold.csv")
    pareada = {
        f"{outro}_menos_{melhor}": comparacao_pareada(scores_por_fold[outro], scores_por_fold[melhor])
        for outro in ORDEM_COMPLEXIDADE if outro != melhor and outro in scores_por_fold
    }
    log(f"[cv] comparação pareada fold a fold: {pareada}")

    # 3. Retreino no treino completo, avaliação única no teste
    t = time.time()
    modelo = _construir(melhor, params).fit(X_tr, y_tr)
    log(f"[final] treinado em {len(X_tr):,} alunos em {time.time() - t:.0f}s")
    proba_te = modelo.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    metricas_teste = calcular_metricas(y_te, pred_te, proba_te)

    baseline = _construir("baseline_prevalencia", params).fit(X_tr, y_tr)
    proba_base = baseline.predict_proba(X_te)[:, 1]
    metricas_baseline = calcular_metricas(y_te, baseline.predict(X_te), proba_base)

    amostra_tr = rng.choice(len(X_tr), size=min(200_000, len(X_tr)), replace=False)
    from sklearn.metrics import roc_auc_score
    auc_treino = round(float(roc_auc_score(y_tr.iloc[amostra_tr],
                                           modelo.predict_proba(X_tr.iloc[amostra_tr])[:, 1])), 4)
    log(f"[teste] modelo: {metricas_teste}")
    log(f"[teste] baseline: ROC-AUC {metricas_baseline['roc_auc']} | acurácia {metricas_baseline['acuracia']}")
    log(f"[teste] ROC-AUC treino {auc_treino} vs teste {metricas_teste['roc_auc']}")

    limiar = analise_limiar(y_te, proba_te)
    limiar.to_csv(REPORTS_DIR / "analise_limiar.csv", index=False)

    # Uso operacional: priorizar os X% de maior risco (capacidade limitada de atendimento).
    priorizacao = analise_priorizacao(y_te, proba_te)
    priorizacao.to_csv(REPORTS_DIR / "analise_priorizacao.csv", index=False)
    top20 = priorizacao.set_index("fracao_priorizada").loc[0.20]
    limiar_risco = float(top20["limiar_p_risco"])
    log(f"[priorização] top 20% de maior risco: {top20['taxa_risco_no_grupo']:.1%} de fato em risco "
        f"(média {1 - y_te.mean():.1%}, lift {top20['lift']}) | cobre {top20['recall_risco']:.1%} dos alunos em risco")

    # Incerteza do resultado no teste: bootstrap reamostrando municípios inteiros
    t = time.time()
    bootstrap = bootstrap_por_grupo(y_te, proba_te, grupos.iloc[idx_teste])
    log(f"[bootstrap] {bootstrap} ({time.time() - t:.0f}s)")

    # Quanto um split aleatório por aluno superestimaria o desempenho (mesmo modelo, mesma subamostra)
    from sklearn.model_selection import StratifiedKFold
    t = time.time()
    auc_aleatorio = cross_validate(_construir(melhor, params), X_sub, y_sub, scoring="roc_auc",
                                   cv=StratifiedKFold(5, shuffle=True, random_state=SEED))["test_score"]
    validacao_comparada = {
        "split_aleatorio_por_aluno": round(float(auc_aleatorio.mean()), 4),
        "split_agrupado_por_municipio": float(comparacao.set_index("modelo").loc[melhor, "cv_roc_auc_medio"]),
    }
    validacao_comparada["otimismo_split_aleatorio"] = round(
        validacao_comparada["split_aleatorio_por_aluno"] - validacao_comparada["split_agrupado_por_municipio"], 4)
    log(f"[validação] aleatório por aluno {validacao_comparada['split_aleatorio_por_aluno']} vs "
        f"agrupado por município {validacao_comparada['split_agrupado_por_municipio']} ({time.time() - t:.0f}s)")

    df_te = df.iloc[idx_teste].assign(proba=proba_te)
    df_te["alfabetizado_int"] = y_te.values
    subgrupos = {}
    for coluna in ("nome_regiao", "rede_label", "sigla_uf"):
        sg = desempenho_por_subgrupo(df_te, coluna, "alfabetizado_int", "proba")
        sg.to_csv(REPORTS_DIR / f"desempenho_por_{coluna}.csv", index=False)
        subgrupos[coluna] = sg

    # Interpretabilidade
    t = time.time()
    perm = importancia_permutacao(modelo, X_te, y_te)
    perm.to_csv(REPORTS_DIR / "importancia_permutacao.csv", index=False)
    shap_vals, X_shap, nomes_shap, metodo_shap = calcular_shap_values(modelo, X_te)
    resumo_shap(shap_vals, nomes_shap).to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    log(f"[interpretabilidade] permutação + SHAP ({metodo_shap}) em {time.time() - t:.0f}s")

    coeficientes = None
    if melhor == "regressao_logistica":
        coeficientes = coeficientes_regressao_logistica(modelo, top_n=100)
        coeficientes["feature"] = coeficientes["feature"].str.split("__", n=1).str[-1]
        coeficientes.to_csv(REPORTS_DIR / "coeficientes_regressao_logistica.csv", index=False)

    # Ablação por grupo de fatores (retreina sem cada grupo / só com ele), mesma CV agrupada
    t = time.time()
    ablacao = ablacao_por_grupo(lambda: _construir(melhor, params), X_sub, y_sub, g_sub, _cv_agrupado(5))
    ablacao.to_csv(REPORTS_DIR / "ablacao_grupos_features.csv", index=False)
    log(f"[ablação] completo {ablacao.attrs['roc_auc_completo']} | "
        f"{ablacao[['grupo', 'perda_ao_remover', 'roc_auc_so_o_grupo']].to_dict(orient='records')} "
        f"({time.time() - t:.0f}s)")

    # Gráficos de EDA e avaliação
    graficos.distribuicao_alvo(df)
    graficos.taxa_por_regiao(df)
    graficos.correlacao_com_alvo(df, FEATURES_NUMERICAS + FEATURES_CONTAGEM)
    graficos.comparacao_modelos(comparacao)
    graficos.matriz_confusao(y_te, pred_te, f"Matriz de confusão — {melhor}\n(teste: municípios nunca vistos)")
    graficos.curva_roc(y_te, proba_te, proba_base, "Curva ROC — teste (municípios nunca vistos)")
    graficos.calibracao(y_te, proba_te)
    graficos.tradeoff_limiar(limiar, limiar_risco)
    graficos.importancia_permutacao(perm)
    graficos.shap_summary(shap_vals, X_shap, nomes_shap)
    graficos.desempenho_regiao(subgrupos["nome_regiao"])
    if coeficientes is not None:
        graficos.coeficientes(coeficientes)
    graficos.ablacao(ablacao)

    # 4. Predições out-of-fold em todos os alunos
    t = time.time()
    proba_oof = cross_val_predict(_construir(melhor, params), X, y, groups=grupos, cv=_cv_agrupado(5),
                                  method="predict_proba", n_jobs=1)[:, 1]
    colunas_oof = ["id_municipio", "id_aluno", "peso_aluno", "nome_municipio", "sigla_uf", "nome_regiao",
                   "rede_label", "meta_ano", COLUNA_ALVO]
    oof = df[colunas_oof].assign(proba_oof=proba_oof)
    oof.to_parquet(PREDICOES_DIR / "predicoes_oof_2024.parquet", index=False)
    from sklearn.metrics import roc_auc_score as _auc
    auc_oof = round(float(_auc(y, proba_oof)), 4)
    log(f"[oof] predições out-of-fold (ROC-AUC {auc_oof}) em {time.time() - t:.0f}s")

    # 5. Modelo final com todos os alunos de 2024 + projeção
    t = time.time()
    modelo_final = _construir(melhor, params).fit(X, y)
    joblib.dump(modelo_final, MODELS_DIR / "modelo_final.joblib")
    proj = montar_dataset_projecao(ANO_PROJECAO, ANO_TREINO)
    proj["proba_projecao"] = modelo_final.predict_proba(proj[TODAS_FEATURES])[:, 1]
    proj.to_parquet(PREDICOES_DIR / f"projecao_{ANO_PROJECAO}.parquet", index=False)
    log(f"[final] modelo com {len(X):,} alunos salvo; projeção {ANO_PROJECAO} para {len(proj):,} alunos "
        f"em {time.time() - t:.0f}s")

    relatorio = {
        "ano_treino": ANO_TREINO,
        "modelo_escolhido": melhor,
        "criterio_escolha": "regra de 1 desvio padrão sobre a ROC-AUC da validação cruzada agrupada por município",
        "justificativa_escolha": justificativa,
        "priorizacao_teste": priorizacao.to_dict(orient="records"),
        "validacao_comparada": validacao_comparada,
        "hiperparametros_hgb": params_hgb,
        "hiperparametros_logistica": params_lr,
        "comparacao_pareada": pareada,
        "bootstrap_teste": bootstrap,
        "ablacao": {"roc_auc_completo": ablacao.attrs["roc_auc_completo"],
                    "grupos": ablacao.to_dict(orient="records")},
        "comparacao_modelos": comparacao.to_dict(orient="records"),
        "metricas_teste": metricas_teste,
        "metricas_baseline_teste": metricas_baseline,
        "roc_auc_treino": auc_treino,
        "roc_auc_oof_todos_alunos": auc_oof,
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "n_municipios_teste": int(grupos.iloc[idx_teste].nunique()),
        "features": TODAS_FEATURES,
        "metodo_shap": metodo_shap,
        "tempo_treino_avaliacao_s": round(time.time() - t0, 1),
    }
    with open(REPORTS_DIR / "metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
    return relatorio


def _indicador_nacional_oficial() -> dict:
    """Indicador Criança Alfabetizada nacional por ano (Gold da Fase 2) — {ano: %}."""
    evolucao = pd.read_parquet(ROOT / "data" / "gold" / "evolucao_temporal")
    brasil = evolucao[evolucao["nivel"] == "brasil"]
    return {int(a): float(v) for a, v in zip(brasil["ano"], brasil["indicador"])}


def rodar_aplicacao_estrategica() -> dict:
    """Ranking de risco, backtest de metas, projeção e clusters — a partir das predições salvas."""
    oof = pd.read_parquet(PREDICOES_DIR / "predicoes_oof_2024.parquet")
    proj = pd.read_parquet(PREDICOES_DIR / f"projecao_{ANO_PROJECAO}.parquet")

    ranking = estrategia.ranking_risco(oof, top_n=50)
    ranking.to_csv(REPORTS_DIR / "ranking_risco_municipios.csv", index=False)
    graficos.ranking_risco(ranking)

    backtest, resumo_backtest = estrategia.backtest_meta(oof)
    backtest.to_csv(REPORTS_DIR / f"backtest_meta_{ANO_TREINO}.csv", index=False)
    graficos.backtest_meta(backtest, resumo_backtest)
    log(f"[estratégia] backtest meta {ANO_TREINO}: {resumo_backtest}")

    projecao, projecao_regiao = estrategia.projecao_meta(proj)
    projecao.to_csv(REPORTS_DIR / f"projecao_meta_{ANO_PROJECAO}.csv", index=False)
    projecao_regiao.to_csv(REPORTS_DIR / f"projecao_meta_{ANO_PROJECAO}_por_regiao.csv", index=False)
    log(f"[estratégia] projeção {ANO_PROJECAO}: {int(projecao.risco_nao_atingir.sum())} de {len(projecao)} "
        f"municípios abaixo da meta")

    # Validação fora do tempo no nível nacional + projeção com nível ajustado (D20)
    oficial = _indicador_nacional_oficial()
    validacao = estrategia.validacao_temporal_nacional(oof, proj, ANO_TREINO, ANO_PROJECAO, oficial)
    log(f"[estratégia] validação temporal nacional: {validacao}")
    projecao_ajustada = regiao_ajustada = None
    if oficial.get(ANO_PROJECAO) is not None:
        proba_aj, delta = estrategia.ajustar_nivel(proj, "proba_projecao", oficial[ANO_PROJECAO])
        projecao_ajustada, regiao_ajustada = estrategia.projecao_meta(proj.assign(proba_projecao=proba_aj))
        projecao_ajustada.to_csv(REPORTS_DIR / f"projecao_meta_{ANO_PROJECAO}_nivel_ajustado.csv", index=False)
        regiao_ajustada.to_csv(REPORTS_DIR / f"projecao_meta_{ANO_PROJECAO}_nivel_ajustado_por_regiao.csv",
                               index=False)
        validacao["deslocamento_logit_ajuste"] = delta
        log(f"[estratégia] projeção {ANO_PROJECAO} com nível ajustado ao oficial ({oficial[ANO_PROJECAO]}%): "
            f"{int(projecao_ajustada.risco_nao_atingir.sum())} de {len(projecao_ajustada)} abaixo da meta")
    graficos.projecao_meta(projecao_regiao, ANO_PROJECAO, regiao_ajustada)
    graficos.validacao_temporal(oficial, validacao, ANO_TREINO, ANO_PROJECAO)

    perfil = estrategia.perfil_municipal(proj)
    clusters, resumo_clusters, composicao, info_clusters = estrategia.clusters_municipais(perfil)
    clusters.to_csv(REPORTS_DIR / "clusters_municipios.csv", index=False)
    resumo_clusters.to_csv(REPORTS_DIR / "clusters_perfil.csv")
    composicao.to_csv(REPORTS_DIR / "clusters_composicao_regiao.csv")
    graficos.clusters_perfil(resumo_clusters, estrategia.COLUNAS_PERFIL)
    graficos.clusters_regiao(composicao)
    log(f"[estratégia] clusters: {info_clusters}")

    resultado = {
        "backtest_meta": resumo_backtest,
        "validacao_temporal_nacional": validacao,
        "projecao": {
            "ano": ANO_PROJECAO,
            "municipios_avaliados": int(len(projecao)),
            "municipios_abaixo_meta": int(projecao.risco_nao_atingir.sum()),
            "por_regiao": projecao_regiao.to_dict(orient="records"),
        },
        "projecao_nivel_ajustado": None if projecao_ajustada is None else {
            "ano": ANO_PROJECAO,
            "taxa_nacional_alvo": oficial[ANO_PROJECAO],
            "municipios_avaliados": int(len(projecao_ajustada)),
            "municipios_abaixo_meta": int(projecao_ajustada.risco_nao_atingir.sum()),
            "por_regiao": regiao_ajustada.to_dict(orient="records"),
        },
        "clusters": {**info_clusters, "perfil": resumo_clusters.reset_index().to_dict(orient="records")},
    }
    with open(REPORTS_DIR / "aplicacao_estrategica.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2, default=str)
    return resultado


def main():
    """Treino completo + aplicação estratégica; `--so-estrategia` refaz só a segunda parte."""
    if "--so-estrategia" not in sys.argv:
        treinar_e_avaliar()
    rodar_aplicacao_estrategica()
    log("\nConcluído.")


if __name__ == "__main__":
    main()
