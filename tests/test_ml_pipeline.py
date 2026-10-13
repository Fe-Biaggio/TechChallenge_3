"""
Testes do pipeline de ML (src/) — dados sintéticos pequenos, não o dataset
real (rodar 1,85M alunos a cada commit seria lento demais). Cobrem as
decisões de modelagem documentadas em docs/decisoes_analiticas.md: fronteira
de vazamento, pipeline treinável com nulos/categorias novas, métricas da
classe de risco e as análises estratégicas.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import aplicacao_estrategica as estrategia
from src.evaluation.metricas import analise_limiar, analise_priorizacao, calcular_metricas, desempenho_por_subgrupo
from src.modeling.pipeline import (
    INDICES_CATEGORICOS_ARVORE,
    pipeline_baseline,
    pipeline_hist_gradient_boosting,
    pipeline_logistica,
    pipeline_random_forest,
    preprocessador_arvore,
)
from src.preprocessing import dataset
from src.preprocessing.dataset import (
    FEATURES_CATEGORICAS,
    FEATURES_CONTAGEM,
    FEATURES_NUMERICAS,
    TODAS_FEATURES,
)


# ─── Fronteira de vazamento ───────────────────────────────────────────────────

COLUNAS_PROIBIDAS = {
    "proficiencia",            # define o alvo por corte determinístico
    "presenca", "preenchimento_caderno",  # só existem no dia da prova
    "serie", "caderno",        # constante / logística de aplicação
    "taxa_alfabetizacao",      # agregado do MESMO ano inclui o próprio aluno
    "indicador_nacional",      # constante dentro de um ano
    "gap_indicador_nacional",  # cópia exata (corr = 1) do indicador municipal
    "faixa_risco",             # discretização redundante do indicador
    "id_escola", "id_aluno", "id_municipio",  # identificadores
    "sigla_uf", "nome_regiao",  # efeito estado-ano aprendido do resultado do próprio ano (choque RS 2024)
}

# Features que são combinação exata de outras — tornam coeficientes não interpretáveis
REDUNDANTES_POR_CONSTRUCAO = {
    "idhm",                     # média geométrica de idhm_educacao/longevidade/renda
    "taxa_rede_municipal_ant",  # = meta_ano − esforco_meta
}


def test_nenhuma_coluna_proibida_entre_as_features():
    assert COLUNAS_PROIBIDAS.isdisjoint(TODAS_FEATURES)


def test_nenhuma_feature_e_combinacao_exata_de_outras():
    assert REDUNDANTES_POR_CONSTRUCAO.isdisjoint(TODAS_FEATURES)


def test_todo_contexto_municipal_e_do_ano_anterior():
    """Feature municipal dinâmica só entra com sufixo _ant (ano anterior) ou é a meta publicada antes."""
    estaticas = {"idhm_educacao", "idhm_longevidade", "idhm_renda", "capital_uf", "amazonia_legal"}
    publicadas_antes = {"meta_ano", "esforco_meta"}
    do_aluno = {"rede_label", "escola_n_alunos"}
    territorio = {"sigla_uf", "nome_regiao"}
    for f in TODAS_FEATURES:
        if f in estaticas | publicadas_antes | do_aluno | territorio:
            continue
        assert f.endswith("_ant"), f"'{f}' parece ser do mesmo ano da prova"


def test_grupos_de_features_sao_disjuntos_e_completos():
    grupos = [set(FEATURES_CATEGORICAS), set(FEATURES_CONTAGEM), set(FEATURES_NUMERICAS)]
    assert sum(len(g) for g in grupos) == len(TODAS_FEATURES)
    assert set().union(*grupos) == set(TODAS_FEATURES)


def test_montar_features_usa_contexto_do_ano_anterior(monkeypatch):
    anos_pedidos = []

    def contexto_falso(ano):
        anos_pedidos.append(ano)
        return pd.DataFrame({"id_municipio": ["1"], "taxa_rede_municipal_ant": [50.0], "meta_ano": [60.0]})

    monkeypatch.setattr(dataset, "_contexto_municipal", contexto_falso)
    monkeypatch.setattr(dataset, "_atributos_estaticos", lambda: pd.DataFrame({"id_municipio": ["1"]}))
    roster = pd.DataFrame({"id_municipio": ["1"] * 3, "id_escola": ["a", "a", "b"], "rede_label": ["Municipal"] * 3})

    df = dataset.montar_features(2024, roster)

    assert anos_pedidos == [2023]
    assert df["esforco_meta"].tolist() == [10.0, 10.0, 10.0]


def test_porte_da_escola_conta_o_roster_inteiro_inclusive_ausentes(monkeypatch):
    monkeypatch.setattr(dataset, "_contexto_municipal",
                        lambda ano: pd.DataFrame({"id_municipio": ["1"], "taxa_rede_municipal_ant": [50.0],
                                                  "meta_ano": [60.0]}))
    monkeypatch.setattr(dataset, "_atributos_estaticos", lambda: pd.DataFrame({"id_municipio": ["1"]}))
    roster = pd.DataFrame({
        "id_municipio": ["1"] * 4, "id_escola": ["a"] * 4, "rede_label": ["Municipal"] * 4,
        "presenca": [True, True, False, False], "preenchimento_caderno": [True] * 4,
    })
    df = dataset.montar_features(2024, roster)
    assert (df["escola_n_alunos"] == 4).all()
    assert len(dataset.populacao_valida(df)) == 2


# ─── Pipelines ───────────────────────────────────────────────────────────────

@pytest.fixture
def dados_sinteticos():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "rede_label": rng.choice(["Municipal", "Estadual"], n),
        "sigla_uf": rng.choice(["SP", "BA", "RS", "CE"], n),
        "nome_regiao": rng.choice(["Sudeste", "Nordeste", "Sul"], n),
        "escola_n_alunos": rng.integers(5, 300, n).astype(float),
        "mun_n_alunos_ant": rng.integers(10, 50_000, n).astype(float),
    })
    for col in FEATURES_NUMERICAS:
        df[col] = rng.normal(size=n)
    df.loc[rng.choice(n, 40, replace=False), "prof_media_ant"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "sigla_uf"] = None
    logit = 1.2 * df["taxa_alfabetizacao_ant"] + 0.8 * df["prof_media_ant"].fillna(0)
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return df[TODAS_FEATURES], y


@pytest.mark.parametrize("construtor", [pipeline_baseline, pipeline_logistica, pipeline_random_forest,
                                        pipeline_hist_gradient_boosting])
def test_todos_os_pipelines_treinam_e_preveem_com_nulos(construtor, dados_sinteticos):
    X, y = dados_sinteticos
    pipe = construtor().fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("construtor", [pipeline_logistica, pipeline_hist_gradient_boosting])
def test_pipeline_aceita_categoria_nunca_vista(construtor, dados_sinteticos):
    X, y = dados_sinteticos
    pipe = construtor().fit(X, y)
    X_novo = X.head(5).copy()
    X_novo["rede_label"] = "Federal"
    assert len(pipe.predict(X_novo)) == 5


def test_categoricas_sao_as_primeiras_colunas_do_preprocessador_de_arvore(dados_sinteticos):
    """O HGB recebe as categóricas por índice — se a ordem mudar, ele trataria número como categoria."""
    X, _ = dados_sinteticos
    nomes = preprocessador_arvore().fit(X).get_feature_names_out()
    for i, col in zip(INDICES_CATEGORICOS_ARVORE, FEATURES_CATEGORICAS):
        assert nomes[i] == f"cat__{col}"


def test_indicador_de_ausencia_e_criado(dados_sinteticos):
    X, _ = dados_sinteticos
    nomes = preprocessador_arvore().fit(X).get_feature_names_out()
    assert "num__missingindicator_prof_media_ant" in nomes


def test_modelo_aprende_sinal_e_supera_baseline(dados_sinteticos):
    from sklearn.metrics import roc_auc_score
    X, y = dados_sinteticos
    auc_modelo = roc_auc_score(y, pipeline_logistica().fit(X, y).predict_proba(X)[:, 1])
    auc_base = roc_auc_score(y, pipeline_baseline().fit(X, y).predict_proba(X)[:, 1])
    assert auc_base == pytest.approx(0.5)
    assert auc_modelo > 0.7


# ─── Métricas ────────────────────────────────────────────────────────────────

def test_metricas_da_classe_de_risco_sao_calculadas_sobre_rotulo_invertido():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])   # acerta 1 alfabetizado, erra 1; acerta os 2 em risco
    proba = np.array([0.9, 0.4, 0.2, 0.1])
    m = calcular_metricas(y_true, y_pred, proba)
    assert m["recall_risco"] == 1.0          # os 2 em risco foram identificados
    assert m["precisao_risco"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["recall"] == 0.5
    assert m["roc_auc"] == 1.0


def test_analise_limiar_tem_colunas_esperadas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    df = analise_limiar(y, rng.uniform(size=500))
    assert {"limiar_p_risco", "precisao_risco", "recall_risco", "f1_risco"} <= set(df.columns)
    assert len(df) <= 50


def test_priorizacao_calcula_lift_sobre_a_taxa_media():
    # 10 alunos, 4 em risco (y=0); o modelo ordena perfeitamente os 4 em risco no topo
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.35, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    df = analise_priorizacao(y, proba, fracoes=(0.2, 0.4)).set_index("fracao_priorizada")
    assert df.loc[0.2, "taxa_risco_no_grupo"] == 1.0
    assert df.loc[0.2, "lift"] == pytest.approx(2.5)        # 100% / 40%
    assert df.loc[0.4, "recall_risco"] == 1.0


def test_regra_um_desvio_padrao_prefere_modelo_mais_simples_em_empate():
    from src.modeling.treinar import escolher_modelo
    comparacao = pd.DataFrame([
        {"modelo": "baseline_prevalencia", "cv_roc_auc_medio": 0.50, "cv_roc_auc_desvio": 0.0},
        {"modelo": "regressao_logistica", "cv_roc_auc_medio": 0.660, "cv_roc_auc_desvio": 0.016},
        {"modelo": "random_forest", "cv_roc_auc_medio": 0.658, "cv_roc_auc_desvio": 0.015},
        {"modelo": "hist_gradient_boosting", "cv_roc_auc_medio": 0.661, "cv_roc_auc_desvio": 0.017},
    ])
    escolhido, _ = escolher_modelo(comparacao)
    assert escolhido == "regressao_logistica"


def test_regra_um_desvio_padrao_mantem_o_melhor_quando_a_diferenca_e_real():
    from src.modeling.treinar import escolher_modelo
    comparacao = pd.DataFrame([
        {"modelo": "regressao_logistica", "cv_roc_auc_medio": 0.62, "cv_roc_auc_desvio": 0.005},
        {"modelo": "random_forest", "cv_roc_auc_medio": 0.64, "cv_roc_auc_desvio": 0.005},
        {"modelo": "hist_gradient_boosting", "cv_roc_auc_medio": 0.70, "cv_roc_auc_desvio": 0.005},
    ])
    escolhido, _ = escolher_modelo(comparacao)
    assert escolhido == "hist_gradient_boosting"


def test_desempenho_por_subgrupo_ignora_grupos_pequenos():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["A"] * 2000 + ["B"] * 50, "y": rng.integers(0, 2, 2050),
                       "p": rng.uniform(size=2050)})
    res = desempenho_por_subgrupo(df, "g", "y", "p", min_n=1000)
    assert res["g"].tolist() == ["A"]


# ─── Aplicação estratégica ───────────────────────────────────────────────────

def _alunos_municipio(id_mun, n, taxa_real, proba, meta, rede="Municipal", regiao="Sul", peso=1.0):
    k = int(round(n * taxa_real))
    return pd.DataFrame({
        "id_municipio": id_mun, "nome_municipio": f"M{id_mun}", "sigla_uf": "RS", "nome_regiao": regiao,
        "rede_label": rede, "peso_aluno": peso, "meta_ano": meta,
        "alfabetizado": [1] * k + [0] * (n - k), "proba_oof": proba, "proba_projecao": proba,
    })


def test_agregacao_municipal_e_ponderada_pelo_peso_amostral():
    df = pd.DataFrame({
        "id_municipio": "1", "nome_municipio": "M1", "sigla_uf": "RS", "nome_regiao": "Sul", "meta_ano": 50.0,
        "peso_aluno": [3.0, 1.0], "alfabetizado": [1, 0], "proba_oof": [0.8, 0.4],
    })
    mun = estrategia.agregar_por_municipio(df, "proba_oof", "alfabetizado")
    assert mun.loc[0, "taxa_real"] == 75.0
    assert mun.loc[0, "taxa_prevista"] == 70.0


def test_backtest_identifica_municipios_abaixo_da_meta():
    df = pd.concat([
        _alunos_municipio("1", 100, 0.40, 0.42, meta=60),  # real abaixo, previsto abaixo -> acerto
        _alunos_municipio("2", 100, 0.80, 0.78, meta=60),  # real acima, previsto acima -> acerto
        _alunos_municipio("3", 100, 0.50, 0.70, meta=60),  # real abaixo, previsto acima -> falso negativo
        _alunos_municipio("4", 100, 0.40, 0.40, meta=60, rede="Estadual"),  # fora do escopo da meta
    ])
    mun, resumo = estrategia.backtest_meta(df)
    assert resumo["municipios_avaliados"] == 3
    assert resumo["real_abaixo_meta"] == 2
    assert resumo["precisao_alerta"] == 1.0
    assert resumo["recall_alerta"] == 0.5


def test_projecao_marca_risco_quando_taxa_projetada_fica_abaixo_da_meta():
    df = pd.concat([
        _alunos_municipio("1", 50, 0.5, 0.55, meta=65, regiao="Norte"),
        _alunos_municipio("2", 50, 0.5, 0.75, meta=65, regiao="Sul"),
    ])
    mun, por_regiao = estrategia.projecao_meta(df)
    risco = dict(zip(mun["id_municipio"], mun["risco_nao_atingir"]))
    assert risco == {"1": True, "2": False}
    assert mun.loc[mun.id_municipio == "1", "gap_projetado_pp"].iloc[0] == pytest.approx(-10.0)
    assert set(por_regiao["nome_regiao"]) == {"Norte", "Sul"}


def test_ranking_exclui_municipios_com_poucos_alunos():
    df = pd.concat([
        _alunos_municipio("grande", 100, 0.3, 0.2, meta=60),
        _alunos_municipio("pequeno", 5, 0.0, 0.01, meta=60),
    ])
    ranking = estrategia.ranking_risco(df)
    assert ranking["id_municipio"].tolist() == ["grande"]


def test_clusters_sao_nomeados_por_taxa_decrescente():
    rng = np.random.default_rng(1)
    blocos = []
    for centro in (30, 60, 90):
        bloco = pd.DataFrame({c: rng.normal(centro, 1, 60) for c in estrategia.COLUNAS_PERFIL})
        bloco["id_municipio"] = [f"{centro}_{i}" for i in range(60)]
        bloco["nome_municipio"] = bloco["id_municipio"]
        bloco["sigla_uf"] = "SP"
        bloco["nome_regiao"] = "Sudeste"
        blocos.append(bloco)
    perfil = pd.concat(blocos, ignore_index=True)
    perfil_out, resumo, _, info = estrategia.clusters_municipais(perfil, k_range=range(2, 5))
    assert info["k_escolhido"] == 3
    taxas = resumo["taxa_alfabetizacao_ant"]
    assert taxas["C1"] > taxas["C2"] > taxas["C3"]


def test_taxa_nacional_usa_so_a_rede_publica_e_o_peso():
    df = pd.DataFrame({
        "rede_label": ["Municipal", "Estadual", "Privada"],
        "peso_aluno": [3.0, 1.0, 100.0],
        "alfabetizado": [1, 0, 0],
    })
    assert estrategia.taxa_nacional(df, "alfabetizado") == 75.0  # privada fora do escopo do indicador oficial


def test_ajuste_de_nivel_bate_a_taxa_alvo_e_preserva_a_ordem():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "rede_label": rng.choice(["Municipal", "Estadual"], 5000),
        "peso_aluno": rng.uniform(0.5, 2.0, 5000),
        "proba_projecao": rng.uniform(0.1, 0.9, 5000),
    })
    ajustada, delta = estrategia.ajustar_nivel(df, "proba_projecao", taxa_alvo_pct=66.0)
    assert np.average(ajustada, weights=df["peso_aluno"]) * 100 == pytest.approx(66.0, abs=0.01)
    assert delta > 0  # média original ~50% -> sobe
    ordem_original = np.argsort(df["proba_projecao"].to_numpy())
    assert (np.diff(ajustada[ordem_original]) >= 0).all()  # ranking não muda


def test_validacao_temporal_reporta_erro_quando_ha_resultado_oficial():
    oof = pd.DataFrame({"rede_label": "Municipal", "peso_aluno": 1.0, "alfabetizado": [1, 0], "proba_oof": [0.6, 0.4]})
    proj = pd.DataFrame({"rede_label": "Municipal", "peso_aluno": 1.0, "proba_projecao": [0.7, 0.5]})
    res = estrategia.validacao_temporal_nacional(oof, proj, 2024, 2025, {2024: 50.0, 2025: 66.0})
    assert res["microdados_ponderado_2024"] == 50.0
    assert res["projetada_2025"] == 60.0
    assert res["erro_projecao_2025_pp"] == -6.0
    sem_2025 = estrategia.validacao_temporal_nacional(oof, proj, 2024, 2025, {2024: 50.0})
    assert "erro_projecao_2025_pp" not in sem_2025
