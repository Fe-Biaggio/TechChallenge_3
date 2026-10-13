"""
Gera a apresentação executiva (PPTX) da Fase 3 — apoio para o vídeo de até
5 minutos pedido no enunciado: o problema educacional, os principais
insights, o valor estratégico da solução e como o modelo apoia políticas
públicas. Roteiro de fala em docs/roteiro_video_fase3.md.

Todos os números vêm dos artefatos gerados por `python -m src.modeling.treinar`
(reports/*.json, reports/*.csv, images/*.png) — nada digitado à mão.
Paleta e componentes visuais seguem o deck da Fase 2 (gerar_apresentacao.py).

Uso:
    python reports/presentation/gerar_apresentacao_fase3.py
"""
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"

METRICAS = json.loads((REPORTS / "metricas_modelo.json").read_text(encoding="utf-8"))
ESTRATEGIA = json.loads((REPORTS / "aplicacao_estrategica.json").read_text(encoding="utf-8"))
PRIORIZACAO = pd.read_csv(REPORTS / "analise_priorizacao.csv").set_index("fracao_priorizada")
PERMUTACAO = pd.read_csv(REPORTS / "importancia_permutacao.csv")
RANKING = pd.read_csv(REPORTS / "ranking_risco_municipios.csv")
PROJ_REGIAO = pd.read_csv(REPORTS / "projecao_meta_2025_por_regiao.csv")
CLUSTERS = pd.read_csv(REPORTS / "clusters_municipios.csv")

# Indicador oficial e meta nacional — camada Gold/Silver da pipeline (Fase 2)
_evolucao = pd.read_parquet(ROOT / "data" / "gold" / "evolucao_temporal")
_evolucao["ano"] = _evolucao["ano"].astype(int)
_brasil = _evolucao[_evolucao["nivel"] == "brasil"].sort_values("ano")
ANO_ULTIMO = int(_brasil["ano"].iloc[-1])
TAXA_NACIONAL_ULTIMA = float(_brasil["indicador"].iloc[-1])
TAXA_NACIONAL_PRIMEIRA, ANO_PRIMEIRO = float(_brasil["indicador"].iloc[0]), int(_brasil["ano"].iloc[0])
_meta_br = pd.read_parquet(ROOT / "data" / "silver" / "meta_brasil")
META_2030 = float(_meta_br[_meta_br["ano_meta"].astype(int) == 2030]["valor_meta"].iloc[0])

NAVY = RGBColor(0x1A, 0x25, 0x3A)
BLUE_MED = RGBColor(0x1F, 0x4E, 0x79)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
RED = RGBColor(0xC0, 0x39, 0x2B)
ORANGE = RGBColor(0xF3, 0x9C, 0x12)
GREEN = RGBColor(0x1E, 0x8B, 0x4C)
BG_LIGHT = RGBColor(0xF4, 0xF6, 0xF9)
TEXT_DARK = RGBColor(0x44, 0x44, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BLUE = RGBColor(0xAA, 0xCC, 0xFF)

FONT = "Calibri"
FOOTER_TEXT = "Tech Challenge Fase 3  |  Predição e Inteligência Analítica para Alfabetização no Brasil  |  Pós-Tech FIAP"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

NOMES_FEATURES = {
    "prof_media_ant": "Proficiência média do município (ano anterior)",
    "prof_p25_ant": "Proficiência dos 25% mais fracos (ano anterior)",
    "prof_desvio_ant": "Desigualdade interna de proficiência",
    "taxa_alfabetizacao_ant": "Taxa de alfabetização do município (ano anterior)",
    "esforco_meta": "Esforço necessário para bater a meta",
    "meta_ano": "Meta do ano",
    "nivel_alfabetizacao_ant": "Nível de alfabetização (ano anterior)",
    "participacao_ant": "Participação na avaliação (ano anterior)",
    "rede_label": "Rede de ensino",
    "escola_n_alunos": "Porte da escola",
    "mun_n_alunos_ant": "Porte do município",
    "idhm_educacao": "IDHM Educação", "idhm_renda": "IDHM Renda",
    "idhm_longevidade": "IDHM Longevidade", "capital_uf": "Capital", "amazonia_legal": "Amazônia Legal",
}


def pct(v, casas=0):
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def num(v, casas=2):
    return f"{v:.{casas}f}".replace(".", ",")


def inteiro(v):
    return f"{int(v):,}".replace(",", ".")


# ─── Helpers visuais ───────────────────────────────────────────────────────────

def novo_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, x, y, w, h, fill=None, line=None, line_width=Pt(1.5), rounded=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if rounded:
        shp.adjustments[0] = 0.06
    if fill is not None:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
    if line is not None:
        shp.line.color.rgb = line
        shp.line.width = line_width
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, size=16, color=TEXT_DARK, bold=False, align=PP_ALIGN.LEFT,
             anchor=None, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    for i, linha in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = linha
        r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, FONT
        r.font.color.rgb = color
    return tb


def add_bullets(slide, x, y, w, h, itens, size=15, cor_marcador=ACCENT, espaco=10):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(itens):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(espaco)
        m = p.add_run()
        m.text = "▶  "
        m.font.size, m.font.bold, m.font.name = Pt(size), True, FONT
        m.font.color.rgb = cor_marcador
        r = p.add_run()
        r.text = item
        r.font.size, r.font.name = Pt(size), FONT
        r.font.color.rgb = TEXT_DARK


def cabecalho(slide, titulo, eyebrow=None):
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=BG_LIGHT)
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.35), fill=NAVY)
    if eyebrow:
        add_text(slide, Inches(0.42), Inches(0.14), Inches(10), Inches(0.3), eyebrow.upper(), size=11,
                 color=LIGHT_BLUE, bold=True)
    add_text(slide, Inches(0.4), Inches(0.42), Inches(12.5), Inches(0.7), titulo, size=28, color=WHITE, bold=True)
    add_rect(slide, 0, Inches(7.15), SLIDE_W, Inches(0.35), fill=NAVY)
    add_text(slide, Inches(0.3), Inches(7.18), Inches(12.7), Inches(0.28), FOOTER_TEXT, size=9, color=LIGHT_BLUE)


def kpi(slide, x, y, w, h, valor, rotulo, cor, detalhe=None):
    add_rect(slide, x, y, w, h, fill=WHITE, line=cor, line_width=Pt(2.25), rounded=True)
    add_text(slide, x, y + Inches(0.15), w, Inches(0.8), valor, size=32, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.12), y + Inches(0.95), w - Inches(0.24), Inches(0.6), rotulo, size=12.5,
             align=PP_ALIGN.CENTER)
    if detalhe:
        add_text(slide, x + Inches(0.12), y + h - Inches(0.38), w - Inches(0.24), Inches(0.3), detalhe, size=10.5,
                 color=cor, bold=True, align=PP_ALIGN.CENTER)


def card(slide, x, y, w, h, titulo, corpo, cor):
    add_rect(slide, x, y, w, h, fill=WHITE, line=cor, line_width=Pt(2), rounded=True)
    add_rect(slide, x, y, Inches(0.09), h, fill=cor)
    add_text(slide, x + Inches(0.28), y + Inches(0.14), w - Inches(0.45), Inches(0.42), titulo, size=15,
             color=NAVY, bold=True)
    add_text(slide, x + Inches(0.28), y + Inches(0.6), w - Inches(0.45), h - Inches(0.7), corpo, size=12)


def imagem(slide, nome, x, y, w=None, h=None):
    caminho = IMAGES / nome
    if caminho.exists():
        slide.shapes.add_picture(str(caminho), x, y, width=w, height=h)


# ─── Números ───────────────────────────────────────────────────────────────────

teste = METRICAS["metricas_teste"]
base = METRICAS["metricas_baseline_teste"]
top20 = PRIORIZACAO.loc[0.20]
backtest = ESTRATEGIA["backtest_meta"]
projecao = ESTRATEGIA["projecao"]
clusters = ESTRATEGIA["clusters"]
ajustada = ESTRATEGIA["projecao_nivel_ajustado"]
validacao = ESTRATEGIA["validacao_temporal_nacional"]
BR_OFICIAL_PROJ = validacao[f"oficial_{projecao['ano']}"]
BR_PROJETADO = validacao[f"projetada_{projecao['ano']}"]
modelo_nome = {"regressao_logistica": "Regressão Logística", "random_forest": "Random Forest",
               "hist_gradient_boosting": "HistGradientBoosting"}[METRICAS["modelo_escolhido"]]
top_fatores = PERMUTACAO.head(3)["feature"].map(lambda f: NOMES_FEATURES.get(f, f)).tolist()
pct_abaixo_2024 = backtest["real_abaixo_meta"] / backtest["municipios_avaliados"]
pct_risco_2025 = projecao["municipios_abaixo_meta"] / projecao["municipios_avaliados"]
# Cluster de maior alfabetização e o caso do Ceará (alta taxa com IDHM Educação abaixo da média)
_melhor = CLUSTERS.groupby("cluster")["taxa_alfabetizacao_ant"].mean().idxmax()
_ce = CLUSTERS[CLUSTERS["sigla_uf"] == "CE"]
CE_PCT_TOPO = (_ce["cluster"] == _melhor).mean()
CE_TAXA = _ce["taxa_alfabetizacao_ant"].mean()
CE_IDHM_E = _ce["idhm_educacao"].mean()
_medias = CLUSTERS.groupby("cluster")[["taxa_alfabetizacao_ant", "idhm_educacao"]].mean()
RICO_IDHM_E = _medias["idhm_educacao"].max()
RICO_TAXA = _medias.loc[_medias["idhm_educacao"].idxmax(), "taxa_alfabetizacao_ant"]

prs = Presentation()
prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

# 1 — Capa
s = novo_slide(prs)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
add_text(s, Inches(1.0), Inches(2.0), Inches(11.3), Inches(1.2), "Quem está em risco de não se alfabetizar?",
         size=44, color=WHITE, bold=True)
add_text(s, Inches(1.0), Inches(3.3), Inches(11.3), Inches(0.8),
         "Predição e inteligência analítica para antecipar o risco e priorizar onde agir", size=22,
         color=LIGHT_BLUE)
add_rect(s, Inches(1.0), Inches(4.35), Inches(4.3), Inches(0.04), fill=ACCENT)
add_text(s, Inches(1.0), Inches(4.6), Inches(11.0), Inches(0.5), "Pós-Tech FIAP  |  1IAST  |  Tech Challenge Fase 3",
         size=16, color=LIGHT_BLUE)

# 2 — O problema
s = novo_slide(prs)
cabecalho(s, f"Avançamos, mas {pct(1 - TAXA_NACIONAL_ULTIMA / 100)} das crianças ainda não se alfabetizam no 2º ano",
          "Problema educacional")
kpi(s, Inches(0.5), Inches(1.7), Inches(3.9), Inches(2.0), inteiro(METRICAS["n_treino"] + METRICAS["n_teste"]),
    "alunos do 2º ano avaliados em 2024 (com prova válida)", ACCENT)
kpi(s, Inches(4.7), Inches(1.7), Inches(3.9), Inches(2.0), pct(TAXA_NACIONAL_ULTIMA / 100, 1),
    f"Indicador Criança Alfabetizada — Brasil, {ANO_ULTIMO} (INEP)", ORANGE,
    f"{pct(TAXA_NACIONAL_PRIMEIRA / 100, 1)} em {ANO_PRIMEIRO} · meta 2030: {pct(META_2030 / 100)}")
kpi(s, Inches(8.9), Inches(1.7), Inches(3.9), Inches(2.0), pct(pct_abaixo_2024, 0),
    "dos municípios (rede municipal) ficaram abaixo da própria meta em 2024", RED)
add_text(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(2.6),
         "O indicador agregado diz onde estamos, mas chega tarde: quando o resultado sai, o ano letivo já acabou.\n"
         "A pergunta desta fase é outra — com o que já se sabe ANTES da prova, dá para antecipar quais alunos e "
         "quais municípios estão em risco, a tempo de agir?",
         size=17, color=NAVY, italic=True)

# 3 — Abordagem
s = novo_slide(prs)
cabecalho(s, "A abordagem: um modelo que só usa o que existe antes da prova", "Solução")
card(s, Inches(0.5), Inches(1.65), Inches(6.0), Inches(2.4), "O que o modelo prevê",
     "Para cada aluno do 2º ano: a probabilidade de atingir o patamar de alfabetização.\n"
     "Somando os alunos de um município, a taxa esperada — comparável com a meta.", ACCENT)
card(s, Inches(6.8), Inches(1.65), Inches(6.0), Inches(2.4), "Com quais informações",
     f"{len(METRICAS['features'])} variáveis da base construída na Fase 2: desempenho do município no ano "
     "anterior, meta do ano, esforço para atingi-la, porte da escola, rede e IDHM. O estado do aluno NÃO entra — ver slide 4.", GREEN)
card(s, Inches(0.5), Inches(4.3), Inches(12.3), Inches(2.5), "A regra que garante que funciona na prática",
     "Nenhuma variável pode depender da prova que estamos tentando prever. Isso exclui a nota do próprio aluno "
     "(que define o resultado) e o indicador municipal do mesmo ano (que inclui o próprio aluno). Tudo que entra "
     "estaria disponível no início do ano letivo — que é quando um gestor precisa da resposta.", ORANGE)

# 4 — Por que confiar
s = novo_slide(prs)
cabecalho(s, "Por que confiar no resultado", "Rigor da solução")
itens = [
    f"Testado em {inteiro(METRICAS['n_municipios_teste'])} municípios que o modelo nunca viu — mede se funciona num "
    "município novo, não se decorou os conhecidos",
    f"Comparado com um palpite trivial: o modelo separa risco de não-risco com AUC {num(teste['roc_auc'])}, "
    f"contra {num(base['roc_auc'])} do palpite",
    f"Escolhido entre 3 algoritmos pela regra mais conservadora: {modelo_nome} — o mais simples entre os que "
    "empataram dentro da margem de erro",
    "Probabilidades calibradas: quando o modelo diz 60%, cerca de 60% de fato se alfabetizam — por isso a soma "
    "por município pode ser comparada com a meta",
    "O estado do aluno ficou de fora: com um só ano de treino, ele aprenderia o choque daquele ano (ex.: enchentes "
    "no RS em 2024) e o contaria duas vezes na projeção do ano seguinte",
]
add_bullets(s, Inches(0.5), Inches(1.7), Inches(7.0), Inches(5.2), itens, size=14)
imagem(s, "07_calibracao.png", Inches(7.8), Inches(1.6), h=Inches(5.3))

# 5 — Principais insights
s = novo_slide(prs)
cabecalho(s, "Principais insights: o que mais pesa no resultado", "Insights")
imagem(s, "09_importancia_permutacao.png", Inches(0.4), Inches(1.55), h=Inches(5.45))
add_text(s, Inches(7.4), Inches(1.7), Inches(5.5), Inches(0.5), "Os três fatores mais decisivos:", size=17,
         color=BLUE_MED, bold=True)
add_bullets(s, Inches(7.4), Inches(2.3), Inches(5.5), Inches(2.6), top_fatores, size=15, cor_marcador=RED)
add_text(s, Inches(7.4), Inches(4.7), Inches(5.5), Inches(2.2),
         "O histórico educacional do município explica mais do que o desenvolvimento socioeconômico amplo (IDHM). "
         "Isso aponta para alavancas de gestão educacional, que respondem em um ano letivo — não só para "
         "indicadores estruturais, que levam uma década para mudar.",
         size=14, color=NAVY, italic=True)

# 6 — Onde agir
s = novo_slide(prs)
cabecalho(s, "Onde agir primeiro: priorização de alunos e municípios", "Valor estratégico")
kpi(s, Inches(0.5), Inches(1.7), Inches(3.7), Inches(2.1), f"{num(top20['lift'], 1)}×",
    "mais alunos em risco no grupo priorizado do que escolhendo ao acaso", GREEN,
    f"top 20%: {pct(top20['taxa_risco_no_grupo'])} em risco")
kpi(s, Inches(0.5), Inches(4.05), Inches(3.7), Inches(2.1), num(backtest["correlacao_prevista_real"]),
    "correlação entre taxa prevista e real por município (2024)", ACCENT,
    f"erro médio {num(backtest['mae_taxa_pp'], 1)} p.p.")
imagem(s, "13_ranking_risco_municipios.png", Inches(4.5), Inches(1.5), h=Inches(5.55))

# 7 — Metas
s = novo_slide(prs)
cabecalho(s, "Antecipar quem não vai bater a meta", "Valor estratégico")
card(s, Inches(0.5), Inches(1.65), Inches(5.6), Inches(2.6), "Teste no passado (2024)",
     f"Com dados de 2023, o modelo apontou {inteiro(backtest['previsto_abaixo_meta'])} municípios abaixo da meta de "
     f"2024. Desses, {pct(backtest['precisao_alerta'])} de fato ficaram abaixo — e o alerta encontrou "
     f"{pct(backtest['recall_alerta'])} de todos os que ficaram.", ACCENT)
card(s, Inches(0.5), Inches(4.45), Inches(5.6), Inches(2.4), f"Projeção {projecao['ano']}: dois cenários",
     f"Entre {inteiro(ajustada['municipios_abaixo_meta'])} e {inteiro(projecao['municipios_abaixo_meta'])} de "
     f"{inteiro(projecao['municipios_avaliados'])} municípios abaixo da meta. O Brasil de {projecao['ano']} já saiu: "
     f"{num(BR_OFICIAL_PROJ, 1)}%, contra {num(BR_PROJETADO, 1)}% "
     "projetados — o modelo acerta a direção e subestima o nível; o cenário ajustado corrige o nível sem mudar a "
     "ordem de prioridade.", RED)
imagem(s, "16_projecao_meta.png", Inches(6.4), Inches(1.7), w=Inches(6.6))

# 8 — Padrões regionais
s = novo_slide(prs)
cabecalho(s, "Municípios parecidos, problemas parecidos", "Padrões regionais")
imagem(s, "14_clusters_perfil.png", Inches(0.4), Inches(1.55), w=Inches(6.6))
imagem(s, "15_clusters_por_regiao.png", Inches(7.1), Inches(1.75), w=Inches(6.0))
add_text(s, Inches(7.1), Inches(4.9), Inches(5.9), Inches(2.1),
         f"{clusters['k_escolhido']} perfis de município. Destaque: {pct(CE_PCT_TOPO)} dos municípios do Ceará estão "
         f"no perfil de maior alfabetização (média {pct(CE_TAXA / 100)}), com IDHM Educação de {num(CE_IDHM_E)} — "
         f"bem abaixo do perfil mais rico ({num(RICO_IDHM_E)}, alfabetização de {pct(RICO_TAXA / 100)}). Contexto "
         "desfavorável não é destino: o estado mantém programa próprio de alfabetização (PAIC) há duas décadas.",
         size=13, color=NAVY, italic=True)

# 9 — Políticas públicas
s = novo_slide(prs)
cabecalho(s, "Como o modelo apoia políticas públicas", "Aplicação")
card(s, Inches(0.5), Inches(1.65), Inches(4.0), Inches(3.4), "1. Priorizar municípios",
     "Ranking de risco antes do ano letivo para direcionar formação de professores, material e "
     "acompanhamento técnico do Compromisso Criança Alfabetizada.", ACCENT)
card(s, Inches(4.7), Inches(1.65), Inches(4.0), Inches(3.4), "2. Alerta de metas",
     "Lista de municípios com risco de não atingir a meta do ano, a tempo de pactuar apoio — "
     "em vez de descobrir depois do resultado.", ORANGE)
card(s, Inches(8.9), Inches(1.65), Inches(4.0), Inches(3.4), "3. Monitorar a cada ano",
     "A pipeline da Fase 2 atualiza a base a cada nova divulgação do INEP; o modelo reavalia o risco "
     "sem trabalho manual.", GREEN)
add_rect(s, Inches(0.5), Inches(5.3), Inches(12.4), Inches(1.5), fill=WHITE, line=RED, line_width=Pt(1.5),
         rounded=True)
add_text(s, Inches(0.8), Inches(5.4), Inches(11.9), Inches(1.3),
         "Uso responsável: o modelo indica onde concentrar APOIO, não quem punir. Uma predição de risco nunca deve "
         "reduzir recursos de um município ou rotular um aluno — o erro individual é alto; o valor está na "
         "priorização agregada.",
         size=13.5, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)

# 10 — Limitações e próximos passos
s = novo_slide(prs)
cabecalho(s, "Limitações e próximos passos", "Conclusão")
add_text(s, Inches(0.5), Inches(1.6), Inches(6.0), Inches(0.5), "O que o modelo não faz", size=18,
         color=RED, bold=True)
add_bullets(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7), [
    "Prever com precisão um aluno específico — faltam dados individuais (frequência, histórico, família)",
    "Garantir que 2025 se comporte como 2024 — só há um par de anos para aprender a relação",
    "Explicar causa — mostra associação, não efeito de intervenção",
], size=14, cor_marcador=RED)
add_text(s, Inches(6.9), Inches(1.6), Inches(6.0), Inches(0.5), "Próximos passos", size=18, color=GREEN, bold=True)
add_bullets(s, Inches(6.9), Inches(2.2), Inches(6.0), Inches(4.7), [
    "Incorporar Censo Escolar (infraestrutura, formação docente) e CadÚnico (vulnerabilidade familiar)",
    "Revalidar a cada nova edição do INEP e acompanhar a precisão do alerta de metas",
    "Piloto com uma secretaria estadual: priorizar municípios e medir o efeito do apoio",
], size=14, cor_marcador=GREEN)

destino = Path(__file__).resolve().parent / "Alfabetizacao_Fase3_Apresentacao.pptx"
prs.save(destino)
print(f"Salvo: {destino} ({len(prs.slides)} slides)")
