"""
Gráficos de EDA e de resultado do modelo — todos salvos como PNG em
reports/figures/, usados pelo README e pelos notebooks.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
CORES = {"True": "#55A868", "False": "#C44E52"}


def _salvar(fig, nome: str) -> Path:
    caminho = FIGURES_DIR / nome
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_distribuicao_alvo(df: pd.DataFrame, coluna_alvo: str = "alfabetizado") -> Path:
    contagem = df[coluna_alvo].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Não alfabetizado", "Alfabetizado"], [contagem.get(False, 0), contagem.get(True, 0)],
           color=["#C44E52", "#55A868"])
    ax.set_title("Distribuição do alvo — alunos 2024")
    ax.set_ylabel("alunos")
    for i, v in enumerate([contagem.get(False, 0), contagem.get(True, 0)]):
        ax.text(i, v, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom")
    return _salvar(fig, "01_distribuicao_alvo.png")


def grafico_taxa_por_regiao(df: pd.DataFrame) -> Path:
    taxa = df.groupby("nome_regiao", observed=True)["alfabetizado"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(taxa.index.astype(str), taxa.values, color="#4C72B0")
    ax.set_xlabel("% alfabetizados")
    ax.set_title("Taxa de alfabetização por região — alunos 2024")
    return _salvar(fig, "02_taxa_por_regiao.png")


def grafico_correlacao_numericas(df: pd.DataFrame, colunas_numericas: list) -> Path:
    corr = df[colunas_numericas + ["alfabetizado"]].astype(float).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlação — features numéricas x alvo")
    return _salvar(fig, "03_correlacao_numericas.png")


def grafico_matriz_confusao(y_true, y_pred, titulo: str = "Matriz de Confusão") -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Não alfabetizado", "Alfabetizado"], ax=ax, cmap="Blues"
    )
    ax.set_title(titulo)
    return _salvar(fig, "04_matriz_confusao.png")


def grafico_curva_roc(y_true, y_proba, titulo: str = "Curva ROC") -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="acaso")
    ax.set_title(titulo)
    ax.legend()
    return _salvar(fig, "05_curva_roc.png")


def grafico_feature_importance(df_importancia: pd.DataFrame, titulo: str, nome_arquivo: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    dados = df_importancia.sort_values(df_importancia.columns[1])
    ax.barh(dados["feature"], dados.iloc[:, 1], color="#4C72B0")
    ax.set_title(titulo)
    return _salvar(fig, nome_arquivo)


def grafico_shap_summary(shap_values, X_transformado, nomes_features, nome_arquivo: str = "07_shap_summary.png") -> Path:
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_transformado, feature_names=nomes_features, show=False)
    fig = plt.gcf()
    caminho = FIGURES_DIR / nome_arquivo
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return caminho
