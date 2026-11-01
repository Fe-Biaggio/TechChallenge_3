"""
Gráficos de EDA, avaliação do modelo e aplicação estratégica — todos
salvos como PNG em images/ (referenciados pelo README e pelos notebooks).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
AZUL, VERDE, VERMELHO, CINZA, LARANJA = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C", "#DD8452"


def _salvar(fig, nome: str) -> Path:
    caminho = IMAGES_DIR / nome
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ─── EDA ─────────────────────────────────────────────────────────────────────

def distribuicao_alvo(df: pd.DataFrame) -> Path:
    contagem = df["alfabetizado"].value_counts()
    valores = [contagem.get(False, 0), contagem.get(True, 0)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Não alfabetizado", "Alfabetizado"], valores, color=[VERMELHO, VERDE])
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)".replace(",", "."), ha="center", va="bottom")
    ax.set_title("Distribuição do alvo — alunos 2024 com prova válida")
    ax.set_ylabel("alunos")
    ax.set_ylim(0, max(valores) * 1.2)
    return _salvar(fig, "01_distribuicao_alvo.png")


def taxa_por_regiao(df: pd.DataFrame) -> Path:
    taxa = df.groupby("nome_regiao", observed=True)["alfabetizado"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(taxa.index.astype(str), taxa.values, color=AZUL)
    for i, v in enumerate(taxa.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("% alfabetizados")
    ax.set_title("Taxa de alfabetização por região — alunos 2024")
    ax.set_xlim(0, 100)
    return _salvar(fig, "02_taxa_por_regiao.png")


def correlacao_com_alvo(df: pd.DataFrame, colunas: list) -> Path:
    corr = df[colunas + ["alfabetizado"]].astype(float).corr()["alfabetizado"].drop("alfabetizado").sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(corr.index, corr.values, color=[VERMELHO if v < 0 else VERDE for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlação de cada feature numérica com 'alfabetizado'")
    ax.set_xlabel("correlação (point-biserial)")
    return _salvar(fig, "03_correlacao_com_alvo.png")


# ─── Avaliação do modelo ─────────────────────────────────────────────────────

def comparacao_modelos(comparacao: pd.DataFrame) -> Path:
    """Barras: ROC-AUC de validação cruzada (média ± desvio) + ROC-AUC de treino (gap = overfitting)."""
    df = comparacao.sort_values("cv_roc_auc_medio")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y - 0.2, df["cv_roc_auc_medio"], 0.4, xerr=df["cv_roc_auc_desvio"], color=AZUL,
            label="validação (CV por município)")
    ax.barh(y + 0.2, df["treino_roc_auc_medio"], 0.4, color=CINZA, label="treino")
    ax.set_yticks(y)
    ax.set_yticklabels(df["modelo"])
    ax.set_xlim(0.45, max(0.8, df["treino_roc_auc_medio"].max() + 0.02))
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Comparação de modelos — validação cruzada agrupada por município")
    ax.legend(loc="lower right")
    return _salvar(fig, "04_comparacao_modelos.png")


def matriz_confusao(y_true, y_pred, titulo: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Não alfab.", "Alfabetizado"], ax=ax, cmap="Blues", values_format=",d"
    )
    ax.set_title(titulo)
    return _salvar(fig, "05_matriz_confusao.png")


def curva_roc(y_true, y_proba, y_proba_baseline, titulo: str) -> Path:
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="modelo final")
    auc_base = roc_auc_score(y_true, y_proba_baseline)
    ax.plot([0, 1], [0, 1], linestyle="--", color=CINZA, label=f"baseline prevalência (AUC = {auc_base:.2f})")
    ax.legend(loc="lower right")
    ax.set_title(titulo)
    return _salvar(fig, "06_curva_roc.png")


def calibracao(y_true, y_proba) -> Path:
    prob_real, prob_prevista = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=CINZA, label="calibração perfeita")
    ax.plot(prob_prevista, prob_real, marker="o", color=AZUL, label="modelo final")
    ax.set_xlabel("probabilidade prevista de estar alfabetizado")
    ax.set_ylabel("fração real de alfabetizados")
    ax.set_title("Calibração — municípios nunca vistos no treino")
    ax.legend()
    return _salvar(fig, "07_calibracao.png")


def tradeoff_limiar(limiar: pd.DataFrame, limiar_triagem: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(limiar["limiar_p_risco"], limiar["precisao_risco"], color=VERMELHO, label="precisão (risco)")
    ax.plot(limiar["limiar_p_risco"], limiar["recall_risco"], color=AZUL, label="recall (risco)")
    ax.axvline(0.5, color=CINZA, linestyle=":", label="limiar padrão (0,50)")
    valor = f"{limiar_triagem:.2f}".replace(".", ",")
    ax.axvline(limiar_triagem, color=LARANJA, linestyle="--", label=f"priorizar top 20% ({valor})")
    ax.set_xlabel("limiar sobre P(não alfabetizado)")
    ax.set_title("Trade-off de limiar para identificar alunos em risco")
    ax.legend()
    return _salvar(fig, "08_tradeoff_limiar.png")


def importancia_permutacao(df: pd.DataFrame) -> Path:
    dados = df.sort_values("queda_roc_auc")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(dados["feature"], dados["queda_roc_auc"], xerr=dados["desvio"], color=AZUL)
    ax.set_xlabel("queda de ROC-AUC ao embaralhar a feature")
    ax.set_title("Importância por permutação — conjunto de teste")
    return _salvar(fig, "09_importancia_permutacao.png")


def shap_summary(shap_values, X_transformado, nomes) -> Path:
    plt.figure()
    shap.summary_plot(shap_values, X_transformado, feature_names=nomes, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP — efeito de cada feature na probabilidade de estar alfabetizado", y=1.02)
    return _salvar(fig, "10_shap_summary.png")


def desempenho_regiao(subgrupos: pd.DataFrame) -> Path:
    df = subgrupos.sort_values("taxa_real")
    y = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ax1.barh(y - 0.2, df["taxa_real"], 0.4, color=VERDE, label="real")
    ax1.barh(y + 0.2, df["taxa_prevista"], 0.4, color=CINZA, label="prevista")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df.iloc[:, 0])
    ax1.set_xlabel("% alfabetizados")
    ax1.set_title("Taxa real vs. prevista")
    ax1.legend()
    ax2.barh(y, df["roc_auc"], color=AZUL)
    ax2.set_xlim(0.5, max(0.8, df["roc_auc"].max() + 0.02))
    ax2.set_xlabel("ROC-AUC")
    ax2.set_title("Poder de discriminação")
    fig.suptitle("Desempenho do modelo por região (teste)")
    return _salvar(fig, "11_desempenho_regiao.png")


# ─── Aplicação estratégica ───────────────────────────────────────────────────

def backtest_meta(mun: pd.DataFrame, resumo: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    cores = np.where(mun["real_abaixo_meta"], VERMELHO, VERDE)
    ax.scatter(mun["taxa_prevista"], mun["taxa_real"], s=6, alpha=0.4, c=cores)
    ax.plot([0, 100], [0, 100], linestyle="--", color=CINZA)
    ax.set_xlabel("taxa prevista (modelo, dados de 2023)")
    ax.set_ylabel("taxa real 2024")
    ax.set_title(f"Backtest 2024 — rede municipal\ncorrelação {resumo['correlacao_prevista_real']:.2f} | "
                 f"erro médio {resumo['mae_taxa_pp']:.1f} p.p.")
    ax.scatter([], [], c=VERMELHO, label="ficou abaixo da meta 2024")
    ax.scatter([], [], c=VERDE, label="atingiu a meta 2024")
    ax.legend(loc="upper left")
    return _salvar(fig, "12_backtest_meta_2024.png")


def ranking_risco(ranking: pd.DataFrame, top_n: int = 20) -> Path:
    df = ranking.head(top_n).iloc[::-1]
    rotulos = df["nome_municipio"].astype(str) + " (" + df["sigla_uf"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(rotulos, df["risco_previsto"], color=VERMELHO)
    ax.scatter(100 - df["taxa_real"], rotulos, color="black", s=12, zorder=3, label="% não alfabetizados real")
    ax.set_xlabel("% previsto de alunos não alfabetizados")
    ax.set_title(f"Top {top_n} municípios de maior risco previsto (≥30 alunos)")
    ax.legend(loc="lower right")
    return _salvar(fig, "13_ranking_risco_municipios.png")


def clusters_perfil(resumo: pd.DataFrame, nomes_colunas: dict) -> Path:
    dados = resumo[list(nomes_colunas)]
    padronizado = (dados - dados.mean()) / dados.std(ddof=0)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(dados) + 2))
    im = ax.imshow(padronizado.values, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(nomes_colunas)))
    ax.set_xticklabels(list(nomes_colunas.values()), rotation=30, ha="right")
    ax.set_yticks(range(len(dados)))
    ax.set_yticklabels([f"{c} (n={int(n)})" for c, n in zip(resumo.index, resumo["n_municipios"])])
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            ax.text(j, i, f"{dados.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="desvio da média (z)")
    ax.set_title("Perfil médio de cada cluster de municípios")
    return _salvar(fig, "14_clusters_perfil.png")


def clusters_regiao(composicao: pd.DataFrame) -> Path:
    dados = composicao.T
    fig, ax = plt.subplots(figsize=(8, 4))
    base = np.zeros(len(dados))
    paleta = plt.get_cmap("RdYlGn_r", len(dados.columns))
    for i, cluster in enumerate(dados.columns):
        ax.barh(dados.index, dados[cluster], left=base, label=cluster, color=paleta(i))
        base += dados[cluster].values
    ax.set_xlabel("% dos municípios da região")
    ax.set_title("Composição de cada região por cluster (C1 = maior alfabetização)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _salvar(fig, "15_clusters_por_regiao.png")


def projecao_meta(por_regiao: pd.DataFrame, ano: int, por_regiao_ajustada: pd.DataFrame = None) -> Path:
    """Barras: % de municípios projetados abaixo da meta, por região — modelo e, se houver, nível ajustado."""
    df = por_regiao.sort_values("pct_em_risco")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if por_regiao_ajustada is None:
        ax.barh(y, df["pct_em_risco"], color=LARANJA)
        pares = [(y, df["pct_em_risco"].to_numpy(), df["municipios"].to_numpy())]
    else:
        aj = por_regiao_ajustada.set_index("nome_regiao").loc[df["nome_regiao"]]
        ax.barh(y + 0.2, df["pct_em_risco"], height=0.4, color=LARANJA, label="modelo (cenário pessimista)")
        ax.barh(y - 0.2, aj["pct_em_risco"], height=0.4, color=AZUL, label="nível ajustado ao Brasil 2025 oficial")
        pares = [(y + 0.2, df["pct_em_risco"].to_numpy(), df["municipios"].to_numpy()),
                 (y - 0.2, aj["pct_em_risco"].to_numpy(), aj["municipios"].to_numpy())]
        ax.legend(loc="lower right", fontsize=8)
    for posicoes, pcts, ns in pares:
        for yy, p, n in zip(posicoes, pcts, ns):
            ax.text(p + 0.5, yy, f"{p:.0f}% de {n}", va="center", fontsize=8)
    ax.set_yticks(y, df["nome_regiao"])
    ax.set_xlabel(f"% de municípios projetados abaixo da meta {ano}")
    ax.set_xlim(0, 100)
    ax.set_title(f"Projeção: municípios em risco de não atingir a meta de {ano} (rede municipal)")
    return _salvar(fig, "16_projecao_meta.png")


def validacao_temporal(oficial: dict, validacao: dict, ano_treino: int, ano_projecao: int) -> Path:
    """Indicador nacional oficial vs. o que o modelo previu (OOF no ano de treino, projeção no seguinte)."""
    anos = sorted(oficial)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(anos, [oficial[a] for a in anos], "-o", color=CINZA, label="Indicador oficial (INEP)")
    ax.scatter([ano_treino], [validacao[f"prevista_oof_{ano_treino}"]], s=90, color=AZUL, zorder=3,
               label=f"previsto out-of-fold ({ano_treino})")
    ax.scatter([ano_projecao], [validacao[f"projetada_{ano_projecao}"]], s=90, color=LARANJA, marker="s", zorder=3,
               label=f"projeção do modelo ({ano_projecao})")
    for a in anos:
        ax.annotate(f"{oficial[a]:.1f}", (a, oficial[a]), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.annotate(f"{validacao[f'projetada_{ano_projecao}']:.1f}", (ano_projecao, validacao[f"projetada_{ano_projecao}"]),
                textcoords="offset points", xytext=(0, -16), ha="center", color=LARANJA)
    ax.set_xticks(anos)
    ax.set_ylabel("% alfabetizados (rede pública)")
    ax.set_title("Validação fora do tempo: Brasil, modelo vs. indicador oficial")
    ax.legend(loc="upper left", fontsize=8)
    return _salvar(fig, "17_validacao_temporal_nacional.png")


def coeficientes(df: pd.DataFrame, top_n: int = 16) -> Path:
    """Coeficientes padronizados da Regressão Logística (efeito no log-odds de estar alfabetizado)."""
    d = df[~df["feature"].str.startswith("missingindicator")].copy()
    d = d.reindex(d["coeficiente"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cores = [VERDE if c > 0 else VERMELHO for c in d["coeficiente"]]
    ax.barh(d["feature"], d["coeficiente"], color=cores)
    ax.axvline(0, color=CINZA, lw=0.8)
    ax.set_xlabel("coeficiente (features padronizadas)\n> 0 aumenta a chance de estar alfabetizado")
    ax.set_title("Regressão Logística — coeficientes (feature importance nativa)")
    return _salvar(fig, "18_coeficientes_logistica.png")


def ablacao(df: pd.DataFrame) -> Path:
    """Perda de ROC-AUC ao retirar cada grupo de fatores, e ROC-AUC só com o grupo."""
    d = df.sort_values("perda_ao_remover")
    completo = df.attrs.get("roc_auc_completo")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].barh(d["grupo"], d["perda_ao_remover"], color=VERMELHO)
    axes[0].set_xlabel("queda de ROC-AUC ao REMOVER o grupo")
    for i, v in enumerate(d["perda_ao_remover"]):
        axes[0].text(max(v, 0), i, f" {v:+.4f}", va="center", fontsize=8)
    axes[1].barh(d["grupo"], d["roc_auc_so_o_grupo"] - 0.5, left=0.5, color=AZUL)
    axes[1].set_xlim(0.5, max(0.7, (completo or 0.65) + 0.02))
    if completo:
        axes[1].axvline(completo, color=CINZA, ls="--", lw=1, label=f"todas as features ({completo:.3f})")
        axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("ROC-AUC usando SÓ o grupo")
    for i, v in enumerate(d["roc_auc_so_o_grupo"]):
        axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.suptitle("Ablação por grupo de fatores (validação cruzada agrupada por município)")
    fig.tight_layout()
    return _salvar(fig, "19_ablacao_grupos.png")
