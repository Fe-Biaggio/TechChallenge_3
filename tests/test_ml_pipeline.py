"""
Testes do pipeline de ML (src/) — usam dados sintéticos pequenos, não o
dataset real (3,8M linhas seria lento demais para rodar a cada commit).
Cobrem especificamente as decisões de modelagem documentadas em
src/preprocessing/dataset.py: ausência de vazamento, pipeline treinável
com nulos/categorias novas, métricas com formato esperado.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metricas import calcular_metricas, validacao_cruzada
from src.modeling.pipeline import construir_pipeline_logistica, construir_pipeline_random_forest
from src.preprocessing.dataset import TODAS_CATEGORICAS, TODAS_FEATURES, TODAS_NUMERICAS


# ─── Regressão contra vazamento de dados ──────────────────────────────────────

def test_proficiencia_nao_esta_nas_features():
    """proficiencia determina 'alfabetizado' por corte determinístico (743 pontos) — nunca pode ser feature."""
    assert "proficiencia" not in TODAS_FEATURES


def test_colunas_administrativas_nao_estao_nas_features():
    """serie (constante), caderno (logística de aplicação), presenca/preenchimento_caderno (usadas só para filtrar a população) não são features."""
    colunas_vazamento_ou_irrelevantes = {"serie", "caderno", "presenca", "preenchimento_caderno", "proficiencia"}
    assert colunas_vazamento_ou_irrelevantes.isdisjoint(set(TODAS_FEATURES))


def test_gap_meta_municipal_excluido_por_cobertura_zero():
    """Confirmado empiricamente (ver dataset.py): fica 100% nulo após o lag de 1 ano, não deve ser feature."""
    assert "gap_meta_municipal" not in TODAS_FEATURES
    assert "atingiu_meta_municipal" not in TODAS_FEATURES


def test_features_numericas_e_categoricas_sao_disjuntas_e_completam_todas_features():
    assert set(TODAS_NUMERICAS).isdisjoint(set(TODAS_CATEGORICAS))
    assert set(TODAS_FEATURES) == set(TODAS_NUMERICAS) | set(TODAS_CATEGORICAS)


# ─── Pipeline treina e prevê em dados sintéticos ──────────────────────────────

@pytest.fixture
def dataset_sintetico():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "taxa_alfabetizacao": rng.uniform(30, 95, n),
        "indicador_nacional": np.full(n, 59.2),
        "gap_indicador_nacional": rng.uniform(-30, 30, n),
        "idhm": rng.uniform(0.4, 0.8, n),
        "idhm_educacao": rng.uniform(0.3, 0.8, n),
        "idhm_longevidade": rng.uniform(0.6, 0.9, n),
        "idhm_renda": rng.uniform(0.4, 0.7, n),
        "faixa_risco": rng.choice(["Crítico", "Atenção", "Adequado", "Meta próxima/atingida"], n),
        "sigla_uf": rng.choice(["SP", "BA", "RS", "CE"], n),
        "nome_regiao": rng.choice(["Sudeste", "Nordeste", "Sul"], n),
        "rede_label": rng.choice(["Municipal", "Estadual"], n),
    })
    # Nulos propositais — simula o ~2% de municípios sem correspondência no lag
    df.loc[rng.choice(n, 10, replace=False), "taxa_alfabetizacao"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "faixa_risco"] = None
    y = pd.Series(rng.integers(0, 2, n), name="alfabetizado")
    return df[TODAS_FEATURES], y


def test_pipeline_logistica_treina_e_preve_com_nulos(dataset_sintetico):
    X, y = dataset_sintetico
    pipe = construir_pipeline_logistica()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert len(pred) == len(X)
    assert proba.shape == (len(X), 2)
    assert set(pred).issubset({0, 1})


def test_pipeline_random_forest_treina_e_preve_com_nulos(dataset_sintetico):
    X, y = dataset_sintetico
    pipe = construir_pipeline_random_forest()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(X)


def test_pipeline_ignora_categoria_nunca_vista_no_treino(dataset_sintetico):
    """OneHotEncoder(handle_unknown='ignore') não deve quebrar com rede 'Federal', nunca vista no treino."""
    X, y = dataset_sintetico
    pipe = construir_pipeline_logistica()
    pipe.fit(X, y)

    X_novo = X.copy()
    X_novo.loc[0, "rede_label"] = "Federal"
    X_novo.loc[1, "sigla_uf"] = "AC"
    pred = pipe.predict(X_novo)  # não deve levantar exceção
    assert len(pred) == len(X_novo)


# ─── Métricas ──────────────────────────────────────────────────────────────

def test_calcular_metricas_formato_esperado():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    y_pred = (y_proba >= 0.5).astype(int)

    m = calcular_metricas(y_true, y_pred, y_proba)
    for chave in ("acuracia", "precisao", "recall", "f1", "roc_auc"):
        assert chave in m
        assert 0.0 <= m[chave] <= 1.0
    assert len(m["matriz_confusao"]) == 2
    assert len(m["matriz_confusao"][0]) == 2


def test_validacao_cruzada_retorna_um_score_por_fold(dataset_sintetico):
    X, y = dataset_sintetico
    pipe = construir_pipeline_logistica()
    resultado = validacao_cruzada(pipe, X, y, n_splits=3)
    assert len(resultado["roc_auc_por_fold"]) == 3
    assert 0.0 <= resultado["roc_auc_medio"] <= 1.0
