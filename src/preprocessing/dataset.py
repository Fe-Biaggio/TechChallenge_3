"""
Monta o dataset de modelagem (nível aluno) a partir da Bronze/Silver/Gold
geradas por pipeline/ (Fase 2) — não reimplementa ingestão/limpeza, só
junta o que já está pronto e aplica as decisões específicas de ML:
população-alvo, remoção de vazamento, e features com defasagem (lag).

Decisões de modelagem (ver README > Tratamento de Data Leakage):

1. População: só alunos com presenca=True e preenchimento_caderno=True.
   `alfabetizado` é derivado da prova; para quem não compareceu ou não
   preencheu o caderno corretamente, o rótulo "False" é administrativo
   (não houve avaliação), não uma medida de proficiência — incluir essas
   linhas ensinaria o modelo a prever ausência/erro de preenchimento, não
   alfabetização.

2. `proficiencia` é excluída das features: é a variável contínua da qual
   `alfabetizado` é derivado por um corte determinístico (743 pontos) —
   usá-la seria prever o rótulo a partir de si mesmo.

3. Features municipais (indicador, meta, gap, faixa de risco) usam o
   **ano anterior** ao da prova do aluno, não o mesmo ano. A tabela Gold
   `indicador_alfabetizacao_municipio` é um agregado que inclui o próprio
   aluno sendo previsto — usar o valor do mesmo ano criaria uma forma sutil
   de vazamento (o indicador "carrega" um pouco do rótulo que queremos
   prever, mais forte em municípios pequenos com poucos alunos avaliados).
   Defasar por um ano quebra esse ciclo e também é mais realista: em
   produção, só o indicador do ano anterior estaria disponível no momento
   de prever o ano corrente.

4. `serie` (sempre "2" — a prova é só para o 2º ano) e `caderno` (código do
   caderno de prova, sorteado por logística de aplicação, não pelo aluno)
   são descartadas por não carregarem nenhum sinal preditivo real.

5. IDHM (Atlas 2010) é constante por município — não há defasagem a
   aplicar, é anterior a qualquer dado de 2023/2024 por construção.

6. O dataset de modelagem usa só alunos de **2024**. Os dados cobrem
   apenas 2023-2024 (ver README > Frequência de atualização da fonte, no
   projeto da Fase 2); alunos de 2023 não têm "ano anterior" disponível
   (não há 2022), então ficariam com as 8 features municipais sempre nulas
   -> sempre imputadas com o mesmo valor constante. Incluir 2023 no treino
   ensinaria o modelo a ignorar essas features (nunca variam lá), e incluir
   em um split temporal treino=2023/teste=2024 faria o modelo treinar sem
   nunca ver as features preenchidas e ser testado só com elas preenchidas
   — um descompasso que inflaria ou mascararia a importância real dessas
   variáveis. Restringir a 2024 (cobertura real ~98% após o lag, ver
   `montar_dataset_modelagem`) evita os dois problemas.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"

COLUNA_ALVO = "alfabetizado"

# Features municipais trazidas da Gold (defasadas em 1 ano — ver docstring).
# gap_meta_municipal/atingiu_meta_municipal ficam de fora: dependem de
# meta_alfabetizacao_2024+ (não existe meta_2023), então ficam 100% nulas
# depois do lag de 1 ano — confirmado empiricamente, não é suposição.
FEATURES_MUNICIPAIS_NUMERICAS = [
    "taxa_alfabetizacao", "indicador_nacional",
    "gap_indicador_nacional", "idhm", "idhm_educacao", "idhm_longevidade", "idhm_renda",
]
FEATURES_MUNICIPAIS_CATEGORICAS = ["faixa_risco", "sigla_uf", "nome_regiao"]

FEATURES_ALUNO_CATEGORICAS = ["rede_label"]

TODAS_NUMERICAS = FEATURES_MUNICIPAIS_NUMERICAS
TODAS_CATEGORICAS = FEATURES_MUNICIPAIS_CATEGORICAS + FEATURES_ALUNO_CATEGORICAS
TODAS_FEATURES = TODAS_NUMERICAS + TODAS_CATEGORICAS


def _ler(camada_dir: Path, nome: str) -> pd.DataFrame:
    caminho = camada_dir / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"'{nome}' não encontrado em {camada_dir} — rode "
            f"`python pipeline/executar_pipeline.py` antes de montar o dataset de modelagem."
        )
    return pq.ParquetDataset(str(caminho)).read().to_pandas()


def _populacao_alunos() -> pd.DataFrame:
    """Filtra para a população-alvo válida (ver decisão 1 na docstring do módulo)."""
    df = _ler(SILVER_DIR, "alunos")
    antes = len(df)
    df = df[(df["presenca"] == True) & (df["preenchimento_caderno"] == True)].copy()  # noqa: E712
    df = df.drop(columns=["serie", "caderno", "proficiencia", "presenca", "preenchimento_caderno"])
    print(f"[dataset] população: {antes:,} alunos -> {len(df):,} com prova válida (presença + caderno preenchido)")
    return df


def _features_municipais_defasadas() -> pd.DataFrame:
    """indicador_alfabetizacao_municipio (Gold), com 'ano' deslocado +1 para virar lag feature."""
    df = _ler(GOLD_DIR, "indicador_alfabetizacao_municipio")
    colunas = ["id_municipio", "ano"] + FEATURES_MUNICIPAIS_NUMERICAS + FEATURES_MUNICIPAIS_CATEGORICAS
    df = df[[c for c in colunas if c in df.columns]].copy()
    # "ano" vem como Categorical (coluna de partição do Parquet) — converte antes de somar
    df["ano"] = df["ano"].astype(int)
    # ano -> ano em que esse dado estará disponível como "ano anterior" de alguém
    df["ano"] = df["ano"] + 1
    df = df.rename(columns={"ano": "ano_predicao"})
    return df


def montar_dataset_modelagem() -> pd.DataFrame:
    """
    Monta o dataset nível-aluno pronto para o pipeline de ML: features +
    coluna alvo (`alfabetizado`), sem nenhuma coluna de vazamento.

    Returns:
        DataFrame com TODAS_FEATURES + COLUNA_ALVO + ['ano'] (mantido só
        para o split temporal train/test, não deve ir ao modelo como feature).
    """
    alunos = _populacao_alunos()
    municipal_lag = _features_municipais_defasadas()

    dataset = alunos.merge(
        municipal_lag,
        left_on=["id_municipio", "ano"],
        right_on=["id_municipio", "ano_predicao"],
        how="left",
    ).drop(columns=["ano_predicao"])

    antes_filtro_ano = len(dataset)
    dataset = dataset[dataset["ano"] == 2024].copy()
    print(
        f"[dataset] restrito a ano=2024 (tem lag real de 2023): "
        f"{antes_filtro_ano:,} -> {len(dataset):,} alunos — "
        f"ver decisão 6 na docstring do módulo (2023 não tem ano anterior disponível)"
    )

    sem_contexto_municipal = dataset["taxa_alfabetizacao"].isna().sum()
    print(
        f"[dataset] {sem_contexto_municipal:,} alunos ({sem_contexto_municipal/len(dataset)*100:.1f}%) "
        f"ainda sem indicador municipal mesmo em 2024 (município sem correspondência na Gold de 2023) — "
        f"tratado via imputação no pipeline de ML, não aqui."
    )

    colunas_finais = TODAS_FEATURES + [COLUNA_ALVO, "ano", "id_municipio", "id_aluno"]
    dataset = dataset[[c for c in colunas_finais if c in dataset.columns]]

    print(f"[dataset] final: {len(dataset):,} linhas x {dataset.shape[1]} colunas")
    print(f"[dataset] taxa de alfabetizados: {dataset[COLUNA_ALVO].mean()*100:.1f}%")
    print(f"[dataset] anos: {sorted(dataset['ano'].unique().tolist())}")

    return dataset


if __name__ == "__main__":
    df = montar_dataset_modelagem()
    print(df.head())
