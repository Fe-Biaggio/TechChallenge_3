"""
Monta o dataset de modelagem (nível aluno) a partir das camadas geradas por
pipeline/ (Fase 2). Não reimplementa ingestão/limpeza — junta o que já está
pronto e aplica as decisões específicas de ML.

Princípio que define a fronteira de vazamento (ver docs/decisoes_analiticas.md):

    Toda feature usada para prever o resultado do ano Y precisa estar
    disponível ANTES da aplicação da prova do ano Y.

Isso decide, de forma verificável, o que entra e o que fica de fora:

  Entra
  - Contexto municipal do ano Y-1: resultado oficial (Gold), resultado da
    rede municipal, nível, participação (metas, vintage Y-1), proficiência
    média/dispersão e porte (microdados de Y-1)
  - Meta do ano Y, publicada em Y-1 (vintage Y-1 da tabela de metas) e o
    "esforço" necessário para atingi-la (meta Y − resultado Y-1)
  - Atributos estáticos do município: capital, Amazônia Legal e os três
    componentes do IDHM 2010
  - Atributos do aluno/escola conhecidos antes da prova: rede de ensino e o
    porte da escola (tamanho da lista de alunos da escola na avaliação)

  Fica de fora
  - proficiencia: o alvo é definido por corte determinístico sobre ela (743)
  - Qualquer agregado do ano Y (o indicador municipal do próprio ano inclui
    o aluno que está sendo previsto)
  - presenca/preenchimento_caderno: só existem no dia da prova — usados
    apenas para definir a população, nunca como feature
  - id_escola como chave entre anos: o código é re-mascarado a cada ano
    (só 865 de ~37 mil pares município+escola se repetem de 2023 para
    2024), então "histórico da escola" ligaria escolas diferentes
  - indicador_nacional / gap_indicador_nacional: dentro de um único ano o
    nacional é constante, e o gap vira uma cópia exata (corr = 1,0) do
    indicador municipal — redundância, não informação
  - faixa_risco: é só uma discretização do indicador municipal (redundante);
    fica para comunicação de resultados, não como feature
  - Redundâncias por construção (combinações exatas de outras features):
    idhm é a média geométrica de idhm_educacao/longevidade/renda (erro
    máx. 0,0005 nos dados) e taxa_rede_municipal_ant = meta_ano −
    esforco_meta. Com as duas, os coeficientes da regressão logística
    deixam de ser interpretáveis (o idhm saía com sinal negativo e os
    componentes positivos, compensando-se). taxa_rede_municipal_ant
    continua sendo calculada — é a base de esforco_meta.
  - serie (constante: só 2º ano) e caderno (logística de aplicação)
  - sigla_uf / nome_regiao como feature: com um único ano de treino, o
    coeficiente de cada estado aprende o choque daquele ano (ex.: enchentes
    no RS em 2024) a partir dos resultados de OUTROS alunos do mesmo
    estado e ano — é o mesmo vazamento por agregado do mesmo ano, no nível
    estadual. Na projeção para o ano seguinte, o choque entraria duas vezes
    (no indicador defasado e no coeficiente). UF e região continuam no
    dataset para agregação e análise, não como entrada do modelo.

População: alunos com presenca=True e preenchimento_caderno=True. Para quem
faltou ou não preencheu a prova, "não alfabetizado" é um rótulo
administrativo, não uma medição.

Ano de treino: 2024. Os dados cobrem só 2023-2024; 2023 não tem ano
anterior (não há 2022) para alimentar as features de contexto.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"

COLUNA_ALVO = "alfabetizado"
COLUNA_GRUPO = "id_municipio"
COLUNA_PESO = "peso_aluno"
ANO_TREINO = 2024

FEATURES_CATEGORICAS = ["rede_label"]
# Contagens com cauda longa — recebem log1p dentro do pipeline de ML
FEATURES_CONTAGEM = ["escola_n_alunos", "mun_n_alunos_ant"]
FEATURES_NUMERICAS = [
    "taxa_alfabetizacao_ant",
    "nivel_alfabetizacao_ant",
    "participacao_ant",
    "meta_ano",
    "esforco_meta",
    "prof_media_ant",
    "prof_desvio_ant",
    "prof_p25_ant",
    "idhm_educacao",
    "idhm_longevidade",
    "idhm_renda",
    "capital_uf",
    "amazonia_legal",
]
TODAS_FEATURES = FEATURES_CATEGORICAS + FEATURES_CONTAGEM + FEATURES_NUMERICAS
COLUNAS_IDENTIFICACAO = ["ano", "id_municipio", "id_escola", "id_aluno", COLUNA_PESO]
# Usadas para agregar e analisar resultados — nunca como entrada do modelo
COLUNAS_TERRITORIO = ["nome_municipio", "sigla_uf", "nome_regiao"]

# Mantidos por compatibilidade com código que importava os nomes antigos
TODAS_CATEGORICAS = FEATURES_CATEGORICAS
TODAS_NUMERICAS = FEATURES_CONTAGEM + FEATURES_NUMERICAS


def _ler(camada_dir: Path, nome: str, colunas: list = None) -> pd.DataFrame:
    caminho = camada_dir / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"'{nome}' não encontrado em {camada_dir} — rode "
            f"`python pipeline/executar_pipeline.py` antes de montar o dataset de modelagem."
        )
    df = pq.ParquetDataset(str(caminho)).read(columns=colunas).to_pandas()
    if "ano" in df.columns:
        df["ano"] = df["ano"].astype(int)
    return df


def carregar_alunos(ano: int) -> pd.DataFrame:
    """Todos os alunos listados na avaliação de `ano` (presentes ou não)."""
    colunas = ["ano", "id_municipio", "id_escola", "id_aluno", "rede_label", "presenca",
               "preenchimento_caderno", "alfabetizado", "proficiencia", COLUNA_PESO]
    df = _ler(SILVER_DIR, "alunos", colunas)
    return df[df["ano"] == ano].copy()


def populacao_valida(alunos: pd.DataFrame) -> pd.DataFrame:
    """Alunos com prova válida — os únicos para os quais `alfabetizado` é uma medição real."""
    mascara = (alunos["presenca"] == True) & (alunos["preenchimento_caderno"] == True)  # noqa: E712
    return alunos[mascara].copy()


def _agregados_microdados(alunos_ano: pd.DataFrame) -> pd.DataFrame:
    """Proficiência média, dispersão, P25 e porte por município, a partir dos microdados de um ano."""
    porte = alunos_ano.groupby("id_municipio").size().rename("mun_n_alunos")
    validos = populacao_valida(alunos_ano)
    prof = validos.groupby("id_municipio")["proficiencia"].agg(
        prof_media="mean", prof_desvio="std", prof_p25=lambda s: s.quantile(0.25)
    )
    return pd.concat([porte, prof], axis=1).reset_index()


def _contexto_municipal(ano: int) -> pd.DataFrame:
    """Contexto municipal de `ano` — usado como 'ano anterior' do ano previsto."""
    gold = _ler(GOLD_DIR, "indicador_alfabetizacao_municipio", ["id_municipio", "ano", "taxa_alfabetizacao"])
    gold = gold[gold["ano"] == ano].drop(columns="ano")

    metas = _ler(BRONZE_DIR, "meta_municipio")
    vintage = metas[metas["ano"] == ano]
    coluna_meta_seguinte = f"meta_alfabetizacao_{ano + 1}"
    vintage = vintage[["id_municipio", "taxa_alfabetizacao", "nivel_alfabetizacao",
                       "percentual_participacao", coluna_meta_seguinte]].rename(columns={
        "taxa_alfabetizacao": "taxa_rede_municipal",
        "nivel_alfabetizacao": "nivel_alfabetizacao",
        "percentual_participacao": "participacao",
        coluna_meta_seguinte: "meta_ano",
    })

    micro = _agregados_microdados(carregar_alunos(ano))

    ctx = gold.merge(vintage, on="id_municipio", how="outer").merge(micro, on="id_municipio", how="outer")
    renomear = {c: f"{c}_ant" for c in ctx.columns if c not in ("id_municipio", "meta_ano")}
    return ctx.rename(columns=renomear)


def _atributos_estaticos() -> pd.DataFrame:
    """UF, região, flags territoriais (diretório) + IDHM 2010 (Atlas)."""
    diretorio = _ler(SILVER_DIR, "diretorio_municipio",
                     ["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "capital_uf", "amazonia_legal"])
    idhm = _ler(SILVER_DIR, "idhm_municipio",
                ["id_municipio", "ano", "idhm", "idhm_educacao", "idhm_longevidade", "idhm_renda"])
    idhm = idhm[idhm["ano"] == idhm["ano"].max()].drop(columns="ano")
    estaticos = diretorio.merge(idhm, on="id_municipio", how="left")
    estaticos["capital_uf"] = estaticos["capital_uf"].fillna(0).astype(float)
    estaticos["amazonia_legal"] = estaticos["amazonia_legal"].astype(float)
    return estaticos


def montar_features(ano_alvo: int, roster: pd.DataFrame) -> pd.DataFrame:
    """
    Junta, para cada aluno de `roster`, as features disponíveis antes da prova
    de `ano_alvo`. `roster` precisa ter id_municipio, id_escola e rede_label;
    colunas extras (alvo, peso, ids) são preservadas.

    `escola_n_alunos` é calculado sobre o roster inteiro recebido — passe a
    lista completa da escola (presentes e ausentes), não só a população válida.
    """
    df = roster.copy()
    df["escola_n_alunos"] = df.groupby("id_escola")["id_escola"].transform("size").astype(float)

    df = df.merge(_contexto_municipal(ano_alvo - 1), on="id_municipio", how="left")
    df = df.merge(_atributos_estaticos(), on="id_municipio", how="left")
    df["esforco_meta"] = df["meta_ano"] - df["taxa_rede_municipal_ant"]
    return df


def montar_dataset_modelagem(ano_alvo: int = ANO_TREINO, verbose: bool = True) -> pd.DataFrame:
    """
    Dataset nível-aluno pronto para o pipeline de ML: features (TODAS_FEATURES),
    alvo, e colunas de identificação (agrupamento por município, peso amostral).
    Nenhuma coluna de vazamento.
    """
    alunos = carregar_alunos(ano_alvo)
    com_features = montar_features(ano_alvo, alunos)
    df = populacao_valida(com_features)

    colunas = TODAS_FEATURES + [COLUNA_ALVO] + COLUNAS_IDENTIFICACAO + COLUNAS_TERRITORIO
    df = df[colunas].reset_index(drop=True)
    df[COLUNA_ALVO] = df[COLUNA_ALVO].astype(bool)

    if verbose:
        print(f"[dataset] ano alvo {ano_alvo}: {len(alunos):,} alunos listados -> {len(df):,} com prova válida")
        print(f"[dataset] {len(TODAS_FEATURES)} features | {df['id_municipio'].nunique():,} municípios "
              f"| taxa de alfabetizados {df[COLUNA_ALVO].mean() * 100:.1f}%")
        nulos = df[TODAS_FEATURES].isna().mean().mul(100).round(1)
        print(f"[dataset] nulos (%): {nulos[nulos > 0].to_dict()}")
    return df


def montar_dataset_projecao(ano_alvo: int, ano_roster: int) -> pd.DataFrame:
    """
    Features para projetar um ano ainda sem microdados (ex.: 2025), usando a
    composição de alunos/escolas de `ano_roster` como aproximação da coorte
    futura. Premissa explícita: rede e porte das escolas mudam pouco de um ano
    para o seguinte — o que muda é o contexto municipal, que vem de ano_alvo-1.
    """
    roster = carregar_alunos(ano_roster)
    com_features = montar_features(ano_alvo, roster)
    df = populacao_valida(com_features)
    colunas = TODAS_FEATURES + COLUNAS_IDENTIFICACAO + COLUNAS_TERRITORIO
    return df[colunas].reset_index(drop=True)


if __name__ == "__main__":
    dados = montar_dataset_modelagem()
    print(dados.head())
