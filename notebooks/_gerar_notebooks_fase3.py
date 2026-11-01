"""
Gera os notebooks 04, 05 e 06 da Fase 3. Os notebooks não re-treinam: leem
os artefatos de `python -m src.modeling.treinar` (reports/, images/, models/).

Uso (da raiz do repositório):
    python notebooks/_gerar_notebooks_fase3.py
    cd notebooks && for nb in 04_eda_modelagem 05_resultados_modelo 06_aplicacao_estrategica; do
        python -m jupyter nbconvert --to notebook --execute --inplace $nb.ipynb; done
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}}


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def salvar(nome, celulas):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = META
    nb["cells"] = celulas
    with open(f"{NB_DIR}/{nome}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("escrito", nome)


SETUP = """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
plt.rcParams["figure.dpi"] = 100"""

# ═════════════════════════════════════════════════════════════════════════════
# 04 — EDA e hipóteses
# ═════════════════════════════════════════════════════════════════════════════
salvar("04_eda_modelagem.ipynb", [
    md("""# 04 — Análise Exploratória e Hipóteses (Fase 3)

Explora o dataset de modelagem montado por `src/preprocessing/dataset.py` a
partir da camada Gold da Fase 2 — 1 linha por aluno de 2024 com prova
válida, 16 features disponíveis **antes** da prova (ver
`docs/decisoes_analiticas.md`, D1) e o alvo `alfabetizado`.

O enunciado pede que a EDA "apoie diretamente as decisões de modelagem e a
construção das hipóteses". Este notebook termina com 5 hipóteses explícitas,
testadas no notebook 05."""),
    code(SETUP + """

from src.preprocessing.dataset import (FEATURES_CONTAGEM, FEATURES_NUMERICAS, TODAS_FEATURES,
                                       montar_dataset_modelagem)

df = montar_dataset_modelagem()
df[TODAS_FEATURES + ["alfabetizado"]].head()"""),

    md("## 1. Alvo e população"),
    code("""print(f"Alunos com prova válida: {len(df):,}")
print(f"Municípios: {df['id_municipio'].nunique():,}")
print(f"Taxa de alfabetizados: {df['alfabetizado'].mean():.1%}")
print(f"Baseline trivial (sempre 'alfabetizado'): acurácia {df['alfabetizado'].mean():.1%}")
display(Image(filename=str(ROOT / "images" / "01_distribuicao_alvo.png")))"""),
    md("""Alvo moderadamente desbalanceado (~60/40). Duas consequências para a
modelagem: (1) acurácia sozinha engana — o baseline trivial já acerta ~60%,
então a métrica de escolha é ROC-AUC e todo resultado é comparado com esse
baseline; (2) não é necessário reamostrar (SMOTE) nem reponderar classes —
o que também preserva a calibração das probabilidades, necessária para
agregá-las por município."""),

    md("## 2. Valores ausentes — e por que não são só ruído"),
    code("""nulos = df[TODAS_FEATURES].isna().mean().mul(100).round(1)
nulos = nulos[nulos > 0].sort_values(ascending=False)
print(nulos.to_string())

sem_micro = df["prof_media_ant"].isna()
print(f"\\nTaxa de alfabetização — com microdados do ano anterior: {df.loc[~sem_micro, 'alfabetizado'].mean():.1%}"
      f" | sem: {df.loc[sem_micro, 'alfabetizado'].mean():.1%}")"""),
    md("""~23% dos alunos estão em municípios sem microdados no ano anterior. Se a
taxa de alfabetização difere entre os dois grupos, a **ausência carrega
informação** — imputar só a mediana a apagaria. Decisão: imputação por
mediana **com indicador de ausência** (`SimpleImputer(add_indicator=True)`),
dentro do pipeline."""),

    md("## 3. Associação entre features numéricas e o alvo"),
    code("""corr = (df[FEATURES_NUMERICAS + FEATURES_CONTAGEM + ["alfabetizado"]].astype(float)
        .corr()["alfabetizado"].drop("alfabetizado").sort_values())
print(corr.round(3).to_string())
display(Image(filename=str(ROOT / "images" / "03_correlacao_com_alvo.png")))"""),
    md("""Leitura: as variáveis de **desempenho educacional do município no ano
anterior** (proficiência média, P25, taxa de alfabetização, nível, meta)
têm as maiores correlações positivas. O **esforço para atingir a meta** é a
mais negativa. Os componentes do **IDHM** — indicador socioeconômico amplo —
aparecem com correlações bem menores. Correlação univariada não controla
pelas outras variáveis; o modelo e a interpretabilidade (notebook 05) dizem
se isso se mantém em conjunto."""),

    md("## 4. Transformação: contagens com cauda longa"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, col in zip(axes, FEATURES_CONTAGEM):
    ax.hist(np.log1p(df[col].dropna()), bins=50, color="#4C72B0")
    ax.set_title(f"log1p({col})")
plt.tight_layout(); plt.show()
print(df[FEATURES_CONTAGEM].describe().round(0).to_string())"""),
    md("""Porte de escola e de município variam em ordens de grandeza (ver mínimo e
máximo na tabela acima) — `log1p` dentro do pipeline estabiliza a escala
para o modelo linear."""),

    md("## 5. Diferenças territoriais"),
    code("""display(Image(filename=str(ROOT / "images" / "02_taxa_por_regiao.png")))
taxa_uf = df.groupby("sigla_uf")["alfabetizado"].mean().sort_values()
print("5 menores:", (taxa_uf.head() * 100).round(1).to_dict())
print("5 maiores:", (taxa_uf.tail() * 100).round(1).to_dict())"""),
    md("""Há diferença regional grande. Mas UF e região **não** entram como
feature: com um único ano de treino, o coeficiente de cada estado aprende o
choque daquele ano (D8 em `docs/decisoes_analiticas.md`) — o modelo tem que
explicar o território pelo **desempenho defasado** e pelo IDHM, que valem
para o ano seguinte."""),

    md("""## 6. Hipóteses analíticas (testadas no notebook 05)

- **H1** — O desempenho educacional do município no ano anterior é o fator
  mais associado ao resultado individual do aluno.
- **H2** — Indicadores socioeconômicos amplos (IDHM) pesam menos que o
  histórico educacional do município.
- **H3** — Quanto maior o esforço necessário para bater a meta do ano, menor
  a chance de o aluno estar alfabetizado.
- **H4** — O modelo é mais útil para **priorizar** (municípios e grupos de
  alunos) do que para prever com certeza um aluno específico — porque não há
  nenhuma informação individual do aluno nos dados.
- **H5** — Existem municípios com alfabetização alta apesar de contexto
  socioeconômico desfavorável — ou seja, o resultado não é determinado pelo
  território."""),
])

# ═════════════════════════════════════════════════════════════════════════════
# 05 — Modelagem e resultados
# ═════════════════════════════════════════════════════════════════════════════
salvar("05_resultados_modelo.ipynb", [
    md("""# 05 — Modelagem, Validação e Interpretação (Fase 3)

Lê os artefatos produzidos por `python -m src.modeling.treinar` — não
re-treina (o treino completo leva ~20 minutos: busca de hiperparâmetros,
comparação de 4 candidatos em validação cruzada agrupada, predições
out-of-fold e modelo final). Todos os números abaixo são lidos dos arquivos,
não digitados."""),
    code(SETUP + """

R = json.loads((ROOT / "reports" / "metricas_modelo.json").read_text(encoding="utf-8"))
print("Modelo escolhido:", R["modelo_escolhido"])
print("Critério:", R["criterio_escolha"])
print("Justificativa:", R["justificativa_escolha"])"""),

    md("""## 1. O pipeline

Um único `Pipeline` do scikit-learn: pré-processamento + modelo. Tudo que é
aprendido (medianas, categorias, escalas) é aprendido só no treino de cada
fold — nada vaza para validação/teste."""),
    code("""import joblib
modelo = joblib.load(ROOT / "models" / "modelo_final.joblib")
modelo"""),

    md("## 2. Desenho da validação"),
    code("""print(f"Treino: {R['n_treino']:,} alunos | Teste: {R['n_teste']:,} alunos de {R['n_municipios_teste']:,} municípios NUNCA vistos")
vc = R["validacao_comparada"]
print(f"\\nMesma validação cruzada, dois desenhos:")
print(f"  split aleatório por aluno:      ROC-AUC {vc['split_aleatorio_por_aluno']}")
print(f"  split agrupado por município:   ROC-AUC {vc['split_agrupado_por_municipio']}")
print(f"  otimismo do split aleatório:    {vc['otimismo_split_aleatorio']:+.4f}")"""),
    md("""O split agrupado por município é o que mede a pergunta real ("funciona num
município novo?"). A diferença entre os dois desenhos é o quanto um split
aleatório por aluno superestimaria o resultado."""),

    md("## 3. Busca de hiperparâmetros e comparação de modelos"),
    code("""tuning = pd.read_csv(ROOT / "reports" / "tuning_hist_gradient_boosting.csv")
print("Melhores configurações do HistGradientBoosting (RandomizedSearchCV, CV agrupada, 3 folds):")
display(tuning.head(5))
tuning_lr = pd.read_csv(ROOT / "reports" / "tuning_regressao_logistica.csv")
print("Regularização da Regressão Logística (GridSearchCV sobre C, CV agrupada, 3 folds):")
display(tuning_lr)
comparacao = pd.DataFrame(R["comparacao_modelos"]).set_index("modelo")
display(comparacao)
display(Image(filename=str(ROOT / "images" / "04_comparacao_modelos.png")))"""),
    md("""Regra de escolha (D14): entre os modelos a até 1 desvio padrão do melhor,
fica o mais simples. Diferenças menores que a variação entre folds não são
evidência de que o modelo mais complexo generaliza melhor — e o gap
treino-validação (coluna `gap_treino_validacao`) mostra quem está mais
perto de overfitting."""),
    code("""por_fold = pd.read_csv(ROOT / "reports" / "comparacao_modelos_por_fold.csv", index_col="fold")
display(por_fold.round(4))
for par, r in R["comparacao_pareada"].items():
    print(f"{par:55s} diferença média {r['diferenca_media']:+.4f}  IC95% {r['ic95_aprox']}  "
          f"(melhor em {r['folds_a_melhor']} de {r['n_folds']} folds)")"""),
    md("""**Comparação pareada.** Os candidatos foram avaliados nos *mesmos* folds,
então dá para comparar fold a fold — o que remove a variação entre grupos de
municípios (mais fáceis ou mais difíceis), que afeta todos os modelos igual.
Se o intervalo de 95% da diferença inclui zero, não há evidência estatística
de que o modelo mais complexo seja melhor; se exclui zero mas a diferença é
pequena (milésimos de AUC), ela não paga o gap de overfitting maior e a perda
de interpretabilidade — é o que a regra de 1 desvio padrão formaliza."""),
    code("""sens = json.loads((ROOT / "reports" / "sensibilidade_modelo.json").read_text(encoding="utf-8"))
display(pd.DataFrame(sens["comparacao_no_teste"]).set_index("modelo"))
print(f"Sobreposição dos 20% de municípios de maior risco entre os dois modelos: {sens['sobreposicao_top20_municipios']:.0%}")
print(f"Correlação entre as taxas municipais previstas pelos dois: {sens['correlacao_taxas_municipais_entre_modelos']}")"""),
    md("""**Sensibilidade (D21).** Checagem posterior à escolha: o HGB treinado no
mesmo treino e avaliado no mesmo teste. Os dois priorizam os mesmos alunos
(lift praticamente idêntico) e quase os mesmos municípios; o HGB é um pouco
melhor no agregado municipal. A escolha não foi revista — mudar de modelo
depois de ver o teste invalidaria o teste — e o HGB fica como **modelo
desafiante** para a próxima rodada com dados novos."""),

    md("## 4. Resultado no teste (municípios nunca vistos)"),
    code("""t, b = R["metricas_teste"], R["metricas_baseline_teste"]
tabela = pd.DataFrame({"modelo": t, "baseline": b}).drop("matriz_confusao")
display(tabela)
print(f"ROC-AUC treino {R['roc_auc_treino']} vs teste {t['roc_auc']} (gap {R['roc_auc_treino'] - t['roc_auc']:.4f})")
print(f"ROC-AUC out-of-fold em TODOS os municípios (5 folds agrupados): {R['roc_auc_oof_todos_alunos']}")
bt = R["bootstrap_teste"]
print(f"IC 95% do ROC-AUC no teste (bootstrap de {bt['n_boot']} reamostragens por {bt['unidade_reamostrada']}): "
      f"{bt['roc_auc_ic95']}")
print(f"IC 95% do lift no top 20%: {bt['lift_top20_ic95']}")"""),
    md("""**Intervalo de confiança.** O bootstrap reamostra **municípios inteiros**,
não alunos — alunos do mesmo município compartilham o contexto e não são
independentes; reamostrá-los um a um daria um intervalo artificialmente
estreito. O intervalo inteiro fica bem acima de 0,5 (baseline) e o do lift
bem acima de 1: o ganho sobre o acaso é estatisticamente robusto."""),
    code("""for img in ["06_curva_roc.png", "05_matriz_confusao.png", "07_calibracao.png"]:
    display(Image(filename=str(ROOT / "images" / img)))"""),
    md("""Calibração: os pontos perto da diagonal significam que uma probabilidade
prevista de 60% corresponde a ~60% de alfabetizados — condição para somar
probabilidades por município e comparar com metas (notebook 06)."""),

    md("## 5. Uso operacional: priorização em vez de limiar fixo"),
    code("""prior = pd.read_csv(ROOT / "reports" / "analise_priorizacao.csv")
display(prior)
display(Image(filename=str(ROOT / "images" / "08_tradeoff_limiar.png")))"""),
    md("""Lida linha a linha: se um programa de reforço só pode atender X% dos
alunos, atendendo os X% de maior risco previsto ele encontra a fração
`taxa_risco_no_grupo` de alunos de fato em risco — `lift` vezes mais do que
escolhendo ao acaso."""),

    md("""## 6. Interpretabilidade e teste das hipóteses

Quatro olhares complementares: **coeficientes** (feature importance nativa do
modelo linear), **importância por permutação** (impacto na métrica, fora da
amostra), **SHAP** (direção e magnitude por aluno) e **ablação por grupo**
(retreina sem cada grupo de fatores — robusta a features correlacionadas)."""),
    code("""coefs = pd.read_csv(ROOT / "reports" / "coeficientes_regressao_logistica.csv")
display(coefs[~coefs.feature.str.startswith("missingindicator")].head(10))
display(Image(filename=str(ROOT / "images" / "18_coeficientes_logistica.png")))"""),
    code("""perm = pd.read_csv(ROOT / "reports" / "importancia_permutacao.csv")
shap_imp = pd.read_csv(ROOT / "reports" / "shap_importance.csv")
display(perm.head(10))
display(Image(filename=str(ROOT / "images" / "09_importancia_permutacao.png")))
display(Image(filename=str(ROOT / "images" / "10_shap_summary.png")))"""),
    code("""ranking = perm.set_index("feature")["queda_roc_auc"]
educacionais = ["taxa_alfabetizacao_ant", "prof_media_ant", "prof_p25_ant", "nivel_alfabetizacao_ant",
                "participacao_ant", "prof_desvio_ant"]
idhm = ["idhm_educacao", "idhm_renda", "idhm_longevidade"]
print("H1 — feature mais importante:", ranking.idxmax(), f"(queda de ROC-AUC {ranking.max():.4f})")
print(f"H2 — soma da importância: históricos educacionais {ranking[educacionais].sum():.4f} | IDHM {ranking[idhm].sum():.4f}")
coef = pd.Series(modelo.named_steps["classificador"].coef_[0] if hasattr(modelo.named_steps["classificador"], "coef_") else [],
                 dtype=float)
if len(coef):
    nomes = modelo.named_steps["preprocessador"].get_feature_names_out()
    coef.index = [n.split("__", 1)[-1] for n in nomes]
    print(f"H3 — coeficiente de esforco_meta (padronizado): {coef['esforco_meta']:+.4f}")"""),
    md("""**H1 — confirmada.** A taxa de alfabetização do município no ano anterior
é, isoladamente, a feature mais importante; a proficiência média do ano
anterior vem em segundo. Juntas respondem pela maior parte da capacidade de
separação do modelo.

**H2 — confirmada, com folga.** A soma da importância dos históricos
educacionais é mais de 20× a dos componentes do IDHM. Isso **não** quer
dizer que condição socioeconômica não importa — ela já está "embutida" no
resultado educacional passado do município. Quer dizer que, para prever o
próximo ano, o histórico educacional carrega quase toda a informação que o
IDHM traria, e mais.

**H3 — confirmada na direção, fraca na magnitude.** O coeficiente de
`esforco_meta` é negativo (mais esforço necessário → menor chance), mas a
contribuição marginal é pequena: a correlação simples de −0,20 vista no
notebook 04 é, em boa parte, explicada pelo próprio resultado passado (quem
precisa de mais esforço é quem teve resultado mais baixo).

Coeficientes da regressão logística são interpretáveis aqui porque as
redundâncias por construção foram removidas (IDHM composto e taxa da rede
municipal — ver D7 em `docs/decisoes_analiticas.md`). Antes disso, o IDHM
saía com coeficiente −1,19 compensando os componentes positivos.

Observação: `mun_n_alunos_ant`, `idhm_renda` e `capital_uf` têm importância
por permutação levemente **negativa** no teste — embaralhá-las melhora a
ROC-AUC em ~0,001–0,003. É um sinal de que não ajudam a generalizar para
municípios novos. Não foram removidas com base nisso, porque seria escolher
features olhando o teste; fica como candidata a seleção de features por
validação cruzada (evolução futura).

**H4** e **H5**: testadas no notebook 06 (agregação municipal e clusters)."""),

    md("""### Ablação por grupo de fatores

A permutação mede uma feature por vez — e subestima um **grupo** quando as
outras features do grupo carregam a mesma informação (ex.: taxa e
proficiência do ano anterior). A ablação retreina o modelo em validação
cruzada agrupada **sem** cada grupo e **só** com ele."""),
    code("""abl = pd.read_csv(ROOT / "reports" / "ablacao_grupos_features.csv")
print("ROC-AUC com todas as features:", R["ablacao"]["roc_auc_completo"])
display(abl)
display(Image(filename=str(ROOT / "images" / "19_ablacao_grupos.png")))"""),
    md("""Leitura: o grupo cuja remoção mais derruba a ROC-AUC é o fator que mais
impacta a predição — e o que, sozinho, chega mais perto do modelo completo é
o que carrega quase toda a informação. É a resposta mais robusta à pergunta
de negócio "quais fatores mais impactam a alfabetização?"."""),

    md("## 7. Onde o modelo funciona pior"),
    code("""for arquivo in ["desempenho_por_nome_regiao.csv", "desempenho_por_rede_label.csv"]:
    display(pd.read_csv(ROOT / "reports" / arquivo))
display(Image(filename=str(ROOT / "images" / "11_desempenho_regiao.png")))"""),
    md("""Diferenças entre taxa real e prevista por região mostram onde o modelo
erra de forma sistemática — como UF/região ficaram fora das features (D8),
um viés regional residual é esperado e é documentado em vez de "corrigido"
com uma variável que não transfere para o ano seguinte."""),
])

# ═════════════════════════════════════════════════════════════════════════════
# 06 — Aplicação estratégica
# ═════════════════════════════════════════════════════════════════════════════
salvar("06_aplicacao_estrategica.ipynb", [
    md("""# 06 — Aplicação Estratégica (Fase 3)

Responde às perguntas de negócio do enunciado com o modelo treinado:

1. Quais municípios apresentam maior risco educacional?
2. Como prever municípios que podem não atingir metas futuras?
3. Quais regiões possuem padrões semelhantes?

(Quais fatores mais impactam / quais variáveis mais influenciam: notebook 05.)

Fonte: `reports/*.csv` e `reports/aplicacao_estrategica.json`, gerados por
`rodar_aplicacao_estrategica()` em `src/modeling/treinar.py`."""),
    code(SETUP + """

E = json.loads((ROOT / "reports" / "aplicacao_estrategica.json").read_text(encoding="utf-8"))"""),

    md("""## 1. Municípios de maior risco educacional

Predições **out-of-fold**: cada município é avaliado por um modelo que nunca
viu nenhum aluno dele. Só municípios com pelo menos 30 alunos avaliados."""),
    code("""ranking = pd.read_csv(ROOT / "reports" / "ranking_risco_municipios.csv", dtype={"id_municipio": str})
display(ranking.head(15)[["nome_municipio", "sigla_uf", "n_alunos", "risco_previsto", "taxa_real"]])
display(Image(filename=str(ROOT / "images" / "13_ranking_risco_municipios.png")))
print("UFs mais frequentes no top 50:", ranking["sigla_uf"].value_counts().head(5).to_dict())"""),

    md("""## 2. Metas futuras

### 2a. Backtest: o modelo teria antecipado quem não bateu a meta de 2024?

Rede municipal (escopo das metas municipais), taxa agregada com o peso
amostral do aluno."""),
    code("""bt = E["backtest_meta"]
for k, v in bt.items():
    print(f"{k:28s} {v}")
display(Image(filename=str(ROOT / "images" / "12_backtest_meta_2024.png")))"""),
    md("""**H4**: compare a correlação município a município (acima) com o ROC-AUC
individual do notebook 05. Errar alunos individualmente e acertar a taxa
do município são compatíveis — os erros individuais se compensam na média.
É no nível agregado que o modelo vira ferramenta de gestão."""),

    md("### 2b. Projeção: quem tende a ficar abaixo da meta de 2025"),
    code("""proj = pd.read_csv(ROOT / "reports" / "projecao_meta_2025.csv", dtype={"id_municipio": str})
por_regiao = pd.read_csv(ROOT / "reports" / "projecao_meta_2025_por_regiao.csv")
display(por_regiao)
display(Image(filename=str(ROOT / "images" / "16_projecao_meta.png")))
print("Maiores gaps projetados (taxa projetada − meta, p.p.):")
display(proj.head(10)[["nome_municipio", "sigla_uf", "taxa_prevista", "meta_ano", "gap_projetado_pp"]])"""),
    code("""# Checagem de plausibilidade: a projeção não pode se afastar muito do último resultado observado
bt_mun = pd.read_csv(ROOT / "reports" / "backtest_meta_2024.csv", dtype={"id_municipio": str})
compara = proj.merge(bt_mun[["id_municipio", "taxa_real"]], on="id_municipio", how="inner")
por_uf = compara.groupby("sigla_uf")[["taxa_real", "taxa_prevista"]].mean().round(1)
por_uf["diferenca"] = (por_uf["taxa_prevista"] - por_uf["taxa_real"]).round(1)
display(por_uf.sort_values("diferenca"))"""),
    md("""Projeção de 2025 vs. resultado real de 2024 por UF. Uma versão anterior
do modelo (com UF como feature) projetava 27,7% para o RS contra 51,6%
reais em 2024 — o choque das enchentes contado duas vezes. Esta checagem é
o que pegou o problema (ver D8 em `docs/decisoes_analiticas.md`)."""),

    md("""### 2c. Validação fora do tempo: o Brasil de 2025 já foi divulgado

Os microdados param em 2024, mas o indicador **nacional** de 2025 já saiu.
É a única medida real do ano projetado — e o teste mais honesto da
projeção."""),
    code("""vt = E["validacao_temporal_nacional"]
for k, v in vt.items():
    print(f"{k:32s} {v}")
display(Image(filename=str(ROOT / "images" / "17_validacao_temporal_nacional.png")))"""),
    md("""Três leituras:

1. **2024, microdados ponderados = oficial**: recalcular a taxa nacional a
   partir dos alunos, com o peso amostral, reproduz o número do INEP — a
   agregação usada em todo o notebook está correta.
2. **2024, previsto out-of-fold ≈ oficial**: somadas, as probabilidades do
   modelo batem a taxa real — a calibração vista no notebook 05 se sustenta
   no agregado.
3. **2025, projetado < oficial**: o modelo acerta a direção (sobe), mas
   subestima a magnitude. Aprendeu a relação de um único par de anos
   (2023→2024) e não antecipa uma aceleração nacional.

Por isso a projeção é apresentada em dois cenários (D20): o modelo puro
(pessimista) e o modelo com o **nível** ajustado ao Brasil 2025 oficial —
todas as probabilidades deslocadas igualmente em logit, o que preserva o
ranking de prioridade e corrige só o patamar."""),
    code("""aj = E["projecao_nivel_ajustado"]
p = E["projecao"]
print(f"Municípios abaixo da meta {p['ano']}: modelo puro {p['municipios_abaixo_meta']:,} de {p['municipios_avaliados']:,} "
      f"| nível ajustado a {aj['taxa_nacional_alvo']}%: {aj['municipios_abaixo_meta']:,}")
display(pd.read_csv(ROOT / "reports" / "projecao_meta_2025_nivel_ajustado_por_regiao.csv"))"""),

    md("""## 3. Regiões com padrões semelhantes

Clusters (KMeans, k escolhido pelo silhouette) sobre o perfil observado do
município: taxa de alfabetização, proficiência média, desigualdade interna,
participação, IDHM Educação e Renda."""),
    code("""c = E["clusters"]
print("silhouette por k:", c["silhouette_por_k"], "-> k =", c["k_escolhido"])
display(pd.DataFrame(c["perfil"]))
display(Image(filename=str(ROOT / "images" / "14_clusters_perfil.png")))
display(Image(filename=str(ROOT / "images" / "15_clusters_por_regiao.png")))"""),
    code("""clusters = pd.read_csv(ROOT / "reports" / "clusters_municipios.csv", dtype={"id_municipio": str})
melhor = clusters.groupby("cluster")["taxa_alfabetizacao_ant"].mean().idxmax()
c1 = clusters[clusters["cluster"] == melhor]
print(f"Cluster de maior alfabetização: {melhor} ({len(c1)} municípios)")
print("UFs mais representadas:", c1["sigla_uf"].value_counts().head(6).to_dict())
pct_uf = clusters.groupby("sigla_uf")["cluster"].apply(lambda s: (s == melhor).mean() * 100).sort_values(ascending=False)
print("\\n% dos municípios de cada UF nesse cluster (top 5):", pct_uf.head().round(1).to_dict())

medias = clusters.groupby("cluster")[["taxa_alfabetizacao_ant", "idhm_educacao", "idhm_renda"]].mean().round(3)
print("\\nMédias por cluster:")
display(medias)
mais_rico = medias["idhm_educacao"].idxmax()
ce = c1[c1["sigla_uf"] == "CE"]
print(f"Ceará no cluster {melhor}: taxa média {ce['taxa_alfabetizacao_ant'].mean():.1f}% | "
      f"IDHM Educação médio {ce['idhm_educacao'].mean():.3f}")
print(f"Cluster de IDHM Educação mais alto ({mais_rico}): taxa média "
      f"{medias.loc[mais_rico, 'taxa_alfabetizacao_ant']:.1f}% | IDHM Educação {medias.loc[mais_rico, 'idhm_educacao']:.3f}")"""),
    md("""**H5**: o cluster de maior alfabetização **não** é o de maior IDHM — tem
IDHM Educação e Renda menores que o cluster intermediário e, mesmo assim,
taxa de alfabetização bem maior (ver médias acima). O caso mais nítido é o
Ceará: quase todos os seus municípios estão no cluster de topo, com taxa
média acima de 90% e IDHM Educação bem abaixo do cluster mais rico. O
estado tem programa próprio de alfabetização há quase duas décadas (PAIC), que serviu de referência para o Compromisso Nacional Criança
Alfabetizada. Os dados não provam causalidade, mas mostram que contexto
socioeconômico desfavorável não é destino — e indicam onde buscar práticas
replicáveis."""),
])
