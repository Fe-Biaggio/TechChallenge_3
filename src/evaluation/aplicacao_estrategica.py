"""
Aplicação estratégica — responde às perguntas de negócio do enunciado da
Fase 3 usando o modelo treinado:

  1. Quais municípios apresentam maior risco educacional?
     -> ranking por risco previsto, com predições OUT-OF-FOLD (cada município
        é avaliado por um modelo que nunca viu nenhum aluno dele)
  2. Como prever municípios que podem não atingir metas futuras?
     -> backtest em 2024 (o modelo, com dados de 2023, teria apontado quem
        não bateu a meta de 2024?) + projeção para a meta de 2025
  3. Quais regiões possuem padrões semelhantes?
     -> clusterização de municípios por perfil educacional/socioeconômico
  4. Quais fatores mais impactam / quais variáveis mais influenciam?
     -> ver src/evaluation/interpretabilidade.py

Agregações municipais usam o peso amostral do aluno (peso_aluno), como o
INEP faz no cálculo oficial da taxa. A comparação com meta usa só a rede
Municipal — é a rede para a qual as metas municipais são definidas.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_ALUNOS_MUNICIPIO = 30


def _media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    pesos = pesos.fillna(1.0)
    return float(np.average(valores.astype(float), weights=pesos)) if pesos.sum() > 0 else np.nan


def agregar_por_municipio(df: pd.DataFrame, coluna_proba: str, coluna_alvo: str = None) -> pd.DataFrame:
    """Taxa prevista (e real, se houver alvo) por município, ponderada pelo peso amostral."""
    def _agg(g):
        linha = {
            "nome_municipio": g["nome_municipio"].iloc[0],
            "sigla_uf": g["sigla_uf"].iloc[0],
            "nome_regiao": g["nome_regiao"].iloc[0],
            "n_alunos": len(g),
            "taxa_prevista": round(_media_ponderada(g[coluna_proba], g["peso_aluno"]) * 100, 2),
            "meta_ano": g["meta_ano"].iloc[0],
        }
        if coluna_alvo:
            linha["taxa_real"] = round(_media_ponderada(g[coluna_alvo], g["peso_aluno"]) * 100, 2)
        return pd.Series(linha)

    return df.groupby("id_municipio").apply(_agg, include_groups=False).reset_index()


def ranking_risco(df_oof: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Municípios com maior risco previsto (100 − taxa prevista), só os com amostra mínima."""
    mun = agregar_por_municipio(df_oof, "proba_oof", "alfabetizado")
    mun = mun[mun["n_alunos"] >= MIN_ALUNOS_MUNICIPIO].copy()
    mun["risco_previsto"] = 100 - mun["taxa_prevista"]
    return mun.sort_values("risco_previsto", ascending=False).head(top_n).reset_index(drop=True)


def backtest_meta(df_oof: pd.DataFrame) -> tuple:
    """
    Para a rede Municipal de 2024: a taxa prevista (out-of-fold, a partir do
    contexto de 2023) indicaria quem ficaria abaixo da meta de 2024?
    """
    mun = agregar_por_municipio(df_oof[df_oof["rede_label"] == "Municipal"], "proba_oof", "alfabetizado")
    mun = mun[(mun["n_alunos"] >= MIN_ALUNOS_MUNICIPIO) & mun["meta_ano"].notna()].copy()
    mun["real_abaixo_meta"] = mun["taxa_real"] < mun["meta_ano"]
    mun["previsto_abaixo_meta"] = mun["taxa_prevista"] < mun["meta_ano"]

    vp = int((mun.real_abaixo_meta & mun.previsto_abaixo_meta).sum())
    fp = int((~mun.real_abaixo_meta & mun.previsto_abaixo_meta).sum())
    fn = int((mun.real_abaixo_meta & ~mun.previsto_abaixo_meta).sum())
    resumo = {
        "municipios_avaliados": int(len(mun)),
        "mae_taxa_pp": round(float((mun.taxa_prevista - mun.taxa_real).abs().mean()), 2),
        "correlacao_prevista_real": round(float(mun[["taxa_prevista", "taxa_real"]].corr().iloc[0, 1]), 4),
        "real_abaixo_meta": int(mun.real_abaixo_meta.sum()),
        "previsto_abaixo_meta": int(mun.previsto_abaixo_meta.sum()),
        "precisao_alerta": round(vp / (vp + fp), 4) if (vp + fp) else None,
        "recall_alerta": round(vp / (vp + fn), 4) if (vp + fn) else None,
        "acuracia_alerta": round(float((mun.real_abaixo_meta == mun.previsto_abaixo_meta).mean()), 4),
    }
    return mun.sort_values("taxa_prevista").reset_index(drop=True), resumo


def projecao_meta(df_proj: pd.DataFrame) -> tuple:
    """Taxa projetada da rede Municipal vs. meta do ano projetado, por município."""
    mun = agregar_por_municipio(df_proj[df_proj["rede_label"] == "Municipal"], "proba_projecao")
    mun = mun[(mun["n_alunos"] >= MIN_ALUNOS_MUNICIPIO) & mun["meta_ano"].notna()].copy()
    mun["gap_projetado_pp"] = (mun["taxa_prevista"] - mun["meta_ano"]).round(2)
    mun["risco_nao_atingir"] = mun["gap_projetado_pp"] < 0
    por_regiao = (mun.groupby("nome_regiao")
                  .agg(municipios=("id_municipio", "count"),
                       em_risco=("risco_nao_atingir", "sum"),
                       gap_medio_pp=("gap_projetado_pp", "mean"))
                  .assign(pct_em_risco=lambda d: (d.em_risco / d.municipios * 100).round(1),
                          gap_medio_pp=lambda d: d.gap_medio_pp.round(2))
                  .sort_values("pct_em_risco", ascending=False)
                  .reset_index())
    return mun.sort_values("gap_projetado_pp").reset_index(drop=True), por_regiao


# Escopo do indicador nacional oficial (Indicador Criança Alfabetizada, rede pública)
REDES_PUBLICAS = ("Municipal", "Estadual")


def taxa_nacional(df: pd.DataFrame, coluna: str) -> float:
    """Taxa nacional da rede pública, ponderada pelo peso amostral do aluno, em %."""
    publica = df[df["rede_label"].isin(REDES_PUBLICAS)]
    return round(_media_ponderada(publica[coluna], publica["peso_aluno"]) * 100, 2)


def ajustar_nivel(df: pd.DataFrame, coluna_proba: str, taxa_alvo_pct: float) -> tuple:
    """
    Desloca todas as probabilidades por um mesmo valor na escala logit até a
    taxa nacional (rede pública, ponderada) bater `taxa_alvo_pct`.

    Corrige só o NÍVEL geral — a correção padrão para mudança de prevalência
    entre o ano de treino e o ano previsto. Preserva a ordem entre alunos e
    municípios (o ranking de risco não muda) e as diferenças relativas.
    Retorna (probabilidades ajustadas, deslocamento em logit).
    """
    p = np.clip(df[coluna_proba].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    publica = df["rede_label"].isin(REDES_PUBLICAS).to_numpy()
    pesos = df["peso_aluno"].fillna(1.0).to_numpy()[publica]
    alvo = taxa_alvo_pct / 100

    baixo, alto = -5.0, 5.0
    for _ in range(60):  # bisseção: a taxa média é monótona no deslocamento
        meio = (baixo + alto) / 2
        taxa = np.average(1 / (1 + np.exp(-(logit[publica] + meio))), weights=pesos)
        baixo, alto = (meio, alto) if taxa < alvo else (baixo, meio)
    delta = (baixo + alto) / 2
    return 1 / (1 + np.exp(-(logit + delta))), round(float(delta), 4)


def validacao_temporal_nacional(df_oof: pd.DataFrame, df_proj: pd.DataFrame, ano_treino: int, ano_projecao: int,
                                oficial: dict) -> dict:
    """
    Compara, no nível nacional, o que o modelo diz com o indicador oficial:
      - ano de treino: taxa dos microdados ponderada (checa a agregação) e
        taxa prevista out-of-fold (checa a calibração agregada)
      - ano projetado: taxa projetada vs. resultado oficial, quando já
        divulgado — validação genuinamente fora do tempo
    `oficial` mapeia ano -> indicador nacional oficial (Gold, evolucao_temporal).
    """
    resultado = {
        f"oficial_{ano_treino}": oficial.get(ano_treino),
        f"microdados_ponderado_{ano_treino}": taxa_nacional(df_oof, "alfabetizado"),
        f"prevista_oof_{ano_treino}": taxa_nacional(df_oof, "proba_oof"),
        f"projetada_{ano_projecao}": taxa_nacional(df_proj, "proba_projecao"),
        f"oficial_{ano_projecao}": oficial.get(ano_projecao),
    }
    if oficial.get(ano_projecao) is not None:
        resultado[f"erro_projecao_{ano_projecao}_pp"] = round(
            resultado[f"projetada_{ano_projecao}"] - oficial[ano_projecao], 2)
    return resultado


COLUNAS_PERFIL = {
    "taxa_alfabetizacao_ant": "Taxa de alfabetização",
    "prof_media_ant": "Proficiência média",
    "prof_desvio_ant": "Desigualdade interna (desvio)",
    "participacao_ant": "Participação na prova",
    "idhm_educacao": "IDHM Educação",
    "idhm_renda": "IDHM Renda",
}


def perfil_municipal(df_proj: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por município com o perfil observado no ano mais recente (colunas *_ant do dataset de projeção)."""
    colunas = ["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao"] + list(COLUNAS_PERFIL)
    return df_proj[colunas].drop_duplicates("id_municipio").dropna().reset_index(drop=True)


def clusters_municipais(perfil: pd.DataFrame, k_range=range(3, 8), seed: int = 42) -> tuple:
    """
    KMeans sobre o perfil padronizado; k escolhido pelo maior silhouette.
    Clusters são nomeados pela ordem da taxa média de alfabetização.
    """
    X = StandardScaler().fit_transform(perfil[list(COLUNAS_PERFIL)])
    silhuetas = {}
    for k in k_range:
        rotulos = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        silhuetas[k] = round(float(silhouette_score(X, rotulos, sample_size=min(4000, len(X)), random_state=seed)), 4)
    k_melhor = max(silhuetas, key=silhuetas.get)
    rotulos = KMeans(n_clusters=k_melhor, n_init=10, random_state=seed).fit_predict(X)

    perfil = perfil.copy()
    perfil["cluster_bruto"] = rotulos
    ordem = (perfil.groupby("cluster_bruto")["taxa_alfabetizacao_ant"].mean()
             .sort_values(ascending=False).index.tolist())
    perfil["cluster"] = perfil["cluster_bruto"].map({c: f"C{i + 1}" for i, c in enumerate(ordem)})
    perfil = perfil.drop(columns="cluster_bruto")

    resumo = (perfil.groupby("cluster")[list(COLUNAS_PERFIL)].mean().round(3)
              .assign(n_municipios=perfil.groupby("cluster").size()))
    composicao_regiao = (pd.crosstab(perfil["cluster"], perfil["nome_regiao"], normalize="columns")
                         .mul(100).round(1))
    return perfil, resumo, composicao_regiao, {"silhouette_por_k": silhuetas, "k_escolhido": k_melhor}
