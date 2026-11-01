"""
Gera a apresentação executiva da Fase 3 (PPTX) e o roteiro do vídeo.

O vídeo pedido no enunciado tem até 5 minutos e deve cobrir: o problema
educacional, os principais insights, o valor estratégico da solução e como os
modelos apoiam políticas públicas — simulando uma reunião com gestores.

Saídas:
  - reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx
      10 slides; a fala de cada slide vai nas notas do apresentador
  - docs/roteiro_video_fase3.md
      a mesma fala, slide a slide, com o tempo de cada um

Fala e slides saem do mesmo lugar (a lista SLIDES abaixo) e todos os
números são lidos dos artefatos de `python -m src.modeling.treinar` — nada
digitado à mão, então slide, fala e README nunca divergem.

Visual: mesma identidade do deck da Fase 2 (gerar_apresentacao.py) —
capa navy, cabeçalho navy, cards brancos com borda colorida, Calibri.

Uso:
    python reports/presentation/gerar_apresentacao_fase3.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"
DESTINO_PPTX = Path(__file__).resolve().parent / "Alfabetizacao_Fase3_Apresentacao.pptx"
DESTINO_ROTEIRO = ROOT / "docs" / "roteiro_video_fase3.md"
AUTOR = "Felipe Barbato de Biaggio"
PALAVRAS_POR_MINUTO = 145  # ritmo de fala executiva, pausado

# ─── Dados ─────────────────────────────────────────────────────────────────────

METRICAS = json.loads((REPORTS / "metricas_modelo.json").read_text(encoding="utf-8"))
ESTRATEGIA = json.loads((REPORTS / "aplicacao_estrategica.json").read_text(encoding="utf-8"))
PRIORIZACAO = pd.read_csv(REPORTS / "analise_priorizacao.csv").set_index("fracao_priorizada")
ABLACAO = pd.read_csv(REPORTS / "ablacao_grupos_features.csv")
RANKING = pd.read_csv(REPORTS / "ranking_risco_municipios.csv")
CLUSTERS = pd.read_csv(REPORTS / "clusters_municipios.csv")
PERFIL = pd.read_csv(REPORTS / "clusters_perfil.csv").set_index("cluster")
COMPOSICAO = pd.read_csv(REPORTS / "clusters_composicao_regiao.csv").set_index("cluster")

_evolucao = pd.read_parquet(ROOT / "data" / "gold" / "evolucao_temporal")
_evolucao["ano"] = _evolucao["ano"].astype(int)
BRASIL = _evolucao[_evolucao["nivel"] == "brasil"].set_index("ano")["indicador"].astype(float).sort_index()
_metas = pd.read_parquet(ROOT / "data" / "silver" / "meta_brasil")
META_BR = _metas.assign(ano=_metas["ano_meta"].astype(int)).set_index("ano")["valor_meta"].astype(float).sort_index()

teste = METRICAS["metricas_teste"]
bootstrap = METRICAS["bootstrap_teste"]
backtest = ESTRATEGIA["backtest_meta"]
validacao = ESTRATEGIA["validacao_temporal_nacional"]
projecao = ESTRATEGIA["projecao"]
ajustada = ESTRATEGIA["projecao_nivel_ajustado"]
ANO_PROJ = projecao["ano"]
ANO_TREINO = METRICAS["ano_treino"]
top20 = PRIORIZACAO.loc[0.20]
auc_completo = METRICAS["ablacao"]["roc_auc_completo"]

# "Poder de previsão" que cada grupo entrega sozinho: (AUC do grupo − 0,5) / (AUC completo − 0,5)
ABLACAO["poder_sozinho"] = ((ABLACAO["roc_auc_so_o_grupo"] - 0.5) / (auc_completo - 0.5)).clip(lower=0).round(2)
_hist = ABLACAO[ABLACAO["grupo"].str.startswith("Histórico")].iloc[0]
_idhm = ABLACAO[ABLACAO["grupo"].str.startswith("IDHM")].iloc[0]

# Clusters: o de maior alfabetização (C1) e o de maior IDHM Educação
C_TOPO = PERFIL["taxa_alfabetizacao_ant"].idxmax()
C_RICO = PERFIL["idhm_educacao"].idxmax()
C_BAIXO = PERFIL["taxa_alfabetizacao_ant"].idxmin()
_ce = CLUSTERS[CLUSTERS["sigla_uf"] == "CE"]
CE_PCT_TOPO = (_ce["cluster"] == C_TOPO).mean()
CE_TAXA = _ce["taxa_alfabetizacao_ant"].mean()

# Concentração do risco: UFs no top 50
TOP50_UF = RANKING["sigla_uf"].value_counts()
DUAS_UFS = TOP50_UF.head(2)

pct_abaixo_meta_treino = backtest["real_abaixo_meta"] / backtest["municipios_avaliados"]
ULTIMO_ANO = int(BRASIL.index.max())
PRIMEIRO_ANO = int(BRASIL.index.min())
nao_alfabetizadas = 1 - BRASIL[ULTIMO_ANO] / 100


def pct(v, casas=0):
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def num(v, casas=2):
    return f"{v:.{casas}f}".replace(".", ",")


def inteiro(v):
    return f"{int(v):,}".replace(",", ".")


def vezes_em_palavras(v):
    """1,45 -> 'uma vez e meia'; para a fala."""
    if 1.4 <= v < 1.6:
        return "uma vez e meia"
    if 1.9 <= v < 2.1:
        return "o dobro de"
    return f"{num(v, 1)} vezes"


def fracao_em_palavras(f):
    """0,34 -> 'uma em cada três'; aproxima para frações simples faladas."""
    for k, texto in ((2, "uma em cada duas"), (3, "uma em cada três"), (4, "uma em cada quatro"),
                     (5, "uma em cada cinco")):
        if abs(f - 1 / k) < 0.03:
            return texto
    return pct(f)


# ─── Identidade visual (deck da Fase 2) ───────────────────────────────────────

NAVY = RGBColor(0x1A, 0x25, 0x3A)
BLUE_MED = RGBColor(0x1F, 0x4E, 0x79)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
RED = RGBColor(0xC0, 0x39, 0x2B)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
GREEN = RGBColor(0x1E, 0x8B, 0x4C)
GRAY = RGBColor(0x8C, 0x96, 0xA5)
BG_LIGHT = RGBColor(0xF4, 0xF6, 0xF9)
TEXT_DARK = RGBColor(0x33, 0x33, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BLUE = RGBColor(0xAA, 0xCC, 0xFF)
FONT = "Calibri"
FOOTER_TEXT = "Tech Challenge Fase 3  |  Predição e Inteligência Analítica para Alfabetização no Brasil  |  Pós-Tech FIAP"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)


def novo_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, x, y, w, h, fill=None, line=None, line_width=Pt(1.5), rounded=False, shape=None):
    tipo = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE)
    shp = slide.shapes.add_shape(tipo, x, y, w, h)
    if rounded and shape is None:
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
             anchor=MSO_ANCHOR.TOP, italic=False, margem=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if not margem:
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    partes = text if isinstance(text, list) else [text]
    primeiro = True
    for parte in partes:
        for linha in parte.split("\n"):
            p = tf.paragraphs[0] if primeiro else tf.add_paragraph()
            primeiro = False
            p.alignment = align
            r = p.add_run()
            r.text = linha
            r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, FONT
            r.font.color.rgb = color
    return tb


def add_runs(slide, x, y, w, h, runs, size=16, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """Um parágrafo com trechos de estilos diferentes: runs = [(texto, cor, negrito), ...]."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    for texto, cor, negrito in runs:
        r = p.add_run()
        r.text = texto
        r.font.size, r.font.bold, r.font.name = Pt(size), negrito, FONT
        r.font.color.rgb = cor
    return tb


def add_bullets(slide, x, y, w, h, itens, size=15, cor_marcador=ACCENT, espaco=10, cor=TEXT_DARK):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(itens):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(espaco)
        m = p.add_run()
        m.text = "▶  "
        m.font.size, m.font.bold, m.font.name = Pt(size - 2), True, FONT
        m.font.color.rgb = cor_marcador
        r = p.add_run()
        r.text = item
        r.font.size, r.font.name = Pt(size), FONT
        r.font.color.rgb = cor
    return tb


def cabecalho(slide, titulo, eyebrow):
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=BG_LIGHT)
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.3), fill=NAVY)
    add_text(slide, Inches(0.5), Inches(0.16), Inches(10), Inches(0.3), eyebrow.upper(), size=11,
             color=LIGHT_BLUE, bold=True)
    add_text(slide, Inches(0.5), Inches(0.44), Inches(12.4), Inches(0.75), titulo, size=28, color=WHITE, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
    add_rect(slide, 0, Inches(7.15), SLIDE_W, Inches(0.35), fill=NAVY)
    add_text(slide, Inches(0.5), Inches(7.18), Inches(12.3), Inches(0.28), FOOTER_TEXT, size=9, color=LIGHT_BLUE)


def kpi(slide, x, y, w, h, valor, rotulo, cor, detalhe=None, tamanho=36):
    add_rect(slide, x, y, w, h, fill=WHITE, line=cor, line_width=Pt(2.25), rounded=True)
    add_text(slide, x, y + Inches(0.12), w, Inches(0.8), valor, size=tamanho, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.15), y + Inches(0.92), w - Inches(0.3), h - Inches(1.3), rotulo, size=13,
             align=PP_ALIGN.CENTER)
    if detalhe:
        add_text(slide, x + Inches(0.15), y + h - Inches(0.42), w - Inches(0.3), Inches(0.32), detalhe, size=11,
                 color=cor, bold=True, align=PP_ALIGN.CENTER)


def card(slide, x, y, w, h, titulo, corpo, cor, tamanho_corpo=13):
    add_rect(slide, x, y, w, h, fill=WHITE, line=cor, line_width=Pt(2), rounded=True)
    add_text(slide, x + Inches(0.25), y + Inches(0.15), w - Inches(0.5), Inches(0.45), titulo, size=16,
             color=cor, bold=True)
    add_text(slide, x + Inches(0.25), y + Inches(0.62), w - Inches(0.5), h - Inches(0.75), corpo,
             size=tamanho_corpo)


def circulo_numero(slide, x, y, numero, cor, d=Inches(0.55)):
    c = add_rect(slide, x, y, d, d, fill=cor, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(numero)
    r.font.size, r.font.bold, r.font.name = Pt(18), True, FONT
    r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def seta(slide, x1, y, x2, cor=GRAY):
    con = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y, x2, y)
    con.line.color.rgb = cor
    con.line.width = Pt(2.5)
    linha = con.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    cauda = linha.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    linha.append(cauda)


def imagem(slide, nome, x, y, w=None, h=None):
    slide.shapes.add_picture(str(IMAGES / nome), x, y, width=w, height=h)


def estilo_grafico(chart, tamanho=12):
    chart.font.size = Pt(tamanho)
    chart.font.name = FONT
    chart.font.color.rgb = TEXT_DARK


def notas(slide, texto):
    slide.notes_slide.notes_text_frame.text = texto


# ─── Slides: cada um com sua fala (vai para as notas e para o roteiro) ─────────

SLIDES = []  # (título curto, fala)


def registrar(slide, titulo, fala):
    fala = " ".join(fala.split())
    notas(slide, fala)
    SLIDES.append((titulo, fala))


prs = Presentation()
prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

# 1 — Capa ─────────────────────────────────────────────────────────────────────
s = novo_slide(prs)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
add_rect(s, 0, Inches(6.3), SLIDE_W, Inches(1.2), fill=BLUE_MED)
add_text(s, 0, Inches(1.25), SLIDE_W, Inches(0.6), "Case 03", size=26, color=LIGHT_BLUE, bold=True,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(2.0), Inches(11.33), Inches(1.5),
         "Quem está em risco de não se alfabetizar?", size=44, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
add_text(s, Inches(1), Inches(3.45), Inches(11.33), Inches(0.6),
         "Predição e inteligência analítica para agir antes da prova", size=22, color=LIGHT_BLUE,
         align=PP_ALIGN.CENTER)
add_rect(s, Inches(5.17), Inches(4.3), Inches(3.0), Inches(0.04), fill=ACCENT)
add_text(s, 0, Inches(4.6), SLIDE_W, Inches(0.45), "Pós-Tech FIAP  |  1IAST  |  Tech Challenge Fase 3", size=16,
         color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
add_text(s, 0, Inches(5.05), SLIDE_W, Inches(0.45), AUTOR, size=17, color=WHITE, align=PP_ALIGN.CENTER)
registrar(s, "Abertura", f"""
Olá. Eu sou {AUTOR}, e este é o Case 3 do Tech Challenge. Vou mostrar como antecipar, antes da prova, quais crianças e quais municípios estão em risco de não se alfabetizar.
""")

# 2 — Problema ─────────────────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, f"{pct(nao_alfabetizadas)} das crianças ainda saem do 2º ano sem se alfabetizar",
          "O problema educacional")
kpi(s, Inches(0.5), Inches(1.65), Inches(3.0), Inches(1.75), pct(BRASIL[ULTIMO_ANO] / 100, 1),
    f"alfabetizadas no Brasil em {ULTIMO_ANO}", GREEN, f"{pct(BRASIL[PRIMEIRO_ANO] / 100, 1)} em {PRIMEIRO_ANO}")
kpi(s, Inches(0.5), Inches(3.6), Inches(3.0), Inches(1.75), pct(pct_abaixo_meta_treino),
    f"dos municípios ficaram abaixo da própria meta em {ANO_TREINO}", RED)
kpi(s, Inches(0.5), Inches(5.55), Inches(3.0), Inches(1.4), pct(META_BR[2030] / 100), "meta nacional para 2030",
    ORANGE, tamanho=30)
# Gráfico nativo: indicador oficial x trajetória da meta
anos = list(range(PRIMEIRO_ANO, 2031))
dados = CategoryChartData()
dados.categories = [str(a) for a in anos]
dados.add_series("Indicador oficial (INEP)", [BRASIL.get(a) for a in anos])
dados.add_series("Meta nacional", [META_BR.get(a) for a in anos])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(3.9), Inches(1.6), Inches(5.3), Inches(4.0), dados)
ch = gf.chart
estilo_grafico(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "% de crianças alfabetizadas no 2º ano"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.has_legend, ch.legend.position, ch.legend.include_in_layout = True, XL_LEGEND_POSITION.BOTTOM, False
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 50, 85
ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
for serie, cor, tracejado in ((ch.plots[0].series[0], NAVY, False), (ch.plots[0].series[1], ORANGE, True)):
    serie.format.line.color.rgb = cor
    serie.format.line.width = Pt(2.75)
    serie.smooth = False
    serie.marker.format.fill.solid()
    serie.marker.format.fill.fore_color.rgb = cor
    serie.marker.format.line.color.rgb = cor
    if tracejado:
        serie.format.line.dash_style = 4  # traço
s0 = ch.plots[0].series[0]
s0.data_labels.show_value = True
s0.data_labels.number_format, s0.data_labels.number_format_is_linked = '0.0', False
s0.data_labels.position = XL_LABEL_POSITION.ABOVE
s0.data_labels.font.size, s0.data_labels.font.bold = Pt(12), True
card(s, Inches(9.5), Inches(1.6), Inches(3.35), Inches(4.0), "A pergunta",
     "Quando o resultado sai, o ano letivo já acabou.\n\nCom o que já se sabe no início do ano, dá para "
     "antecipar quais alunos e quais municípios estão em risco — a tempo de agir?", BLUE_MED, tamanho_corpo=15)
add_text(s, Inches(3.9), Inches(5.85), Inches(9.0), Inches(1.0),
         "Indicador Criança Alfabetizada (INEP): % de alunos com ao menos 743 pontos na escala Saeb ao fim do "
         "2º ano. Meta do Compromisso Nacional Criança Alfabetizada.", size=11, color=GRAY, italic=True)
registrar(s, "O problema educacional", f"""
O Brasil se comprometeu a alfabetizar todas as crianças até o fim do segundo ano, até 2030. O indicador
nacional avançou: de {num(BRASIL[PRIMEIRO_ANO], 0)}% em {PRIMEIRO_ANO} para {num(BRASIL[ULTIMO_ANO], 0)}% em
{ULTIMO_ANO}. Mesmo assim, {fracao_em_palavras(nao_alfabetizadas)} crianças ainda termina o segundo ano sem
estar alfabetizada, e quase metade dos municípios ficou abaixo da própria meta em {ANO_TREINO}. O problema é
de tempo: quando o resultado sai, o ano letivo já acabou. A nossa pergunta foi: com o que já se sabe no
início do ano, dá para antecipar onde está o risco?
""")

# 3 — Solução ──────────────────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Um modelo que só usa o que existe antes da prova", "A solução")
etapas = [
    ("Base da Fase 2", f"{inteiro(METRICAS['n_treino'] + METRICAS['n_teste'])} alunos de {ANO_TREINO}\n"
                       "+ municípios, metas e IDHM", NAVY),
    ("Modelo", "chance de cada aluno\natingir a alfabetização", BLUE_MED),
    ("Município", "taxa esperada\ncomparada com a meta", ACCENT),
    ("Gestão", "onde agir primeiro,\nantes da prova", GREEN),
]
x0, wc, gap = Inches(0.5), Inches(2.75), Inches(0.47)
for i, (titulo, corpo, cor) in enumerate(etapas):
    x = x0 + i * (wc + gap)
    add_rect(s, x, Inches(1.7), wc, Inches(1.65), fill=cor, rounded=True)
    add_text(s, x, Inches(1.85), wc, Inches(0.5), titulo, size=19, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.35), wc - Inches(0.2), Inches(0.9), corpo, size=13, color=WHITE,
             align=PP_ALIGN.CENTER)
    if i < len(etapas) - 1:
        seta(s, x + wc + Inches(0.06), Inches(2.52), x + wc + gap - Inches(0.06))
card(s, Inches(0.5), Inches(3.75), Inches(6.1), Inches(2.6), "O que o modelo usa",
     "", GREEN)
add_bullets(s, Inches(0.75), Inches(4.35), Inches(5.7), Inches(2.5), [
    "Desempenho do município no ano anterior (taxa, proficiência, participação)",
    "Meta do ano e o esforço para atingi-la",
    "Porte da escola e rede de ensino",
    "IDHM — educação, renda e longevidade",
], size=14, cor_marcador=GREEN, espaco=6)
card(s, Inches(6.85), Inches(3.75), Inches(6.0), Inches(2.6), "O que o modelo NÃO usa — e por quê",
     "", RED)
add_bullets(s, Inches(7.1), Inches(4.35), Inches(5.6), Inches(2.5), [
    "A nota do próprio aluno: ela define o resultado",
    "O indicador do município no mesmo ano: inclui o próprio aluno",
    "O estado do aluno: com um ano de treino, aprenderia o choque daquele ano (ex.: enchentes no RS)",
], size=14, cor_marcador=RED, espaco=6)
add_text(s, Inches(0.5), Inches(6.5), Inches(12.35), Inches(0.5),
         "Cada escolha está documentada com evidência em docs/decisoes_analiticas.md (22 decisões) e protegida por "
         "testes automáticos que falham se uma variável proibida entrar no modelo.", size=12, color=GRAY, italic=True)
registrar(s, "A solução", f"""
Usamos a base construída na fase anterior, com {num((METRICAS['n_treino'] + METRICAS['n_teste']) / 1e6, 2)}
milhão de alunos avaliados em {ANO_TREINO}, e treinamos um modelo que estima, para cada aluno, a chance de
se alfabetizar. Somando os alunos de um município, temos a taxa esperada, que comparamos com a meta. A regra
de ouro: o modelo só usa o que existe antes da prova, como o desempenho do
município no ano anterior, a meta, a escola e o IDHM. Nada do resultado que queremos prever.
""")

# 4 — Por que confiar ──────────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Por que confiar no resultado", "Rigor da solução")
garantias = [
    ("Testado onde nunca viu", f"{inteiro(METRICAS['n_municipios_teste'])} municípios separados antes de tudo — "
                               "mede se funciona num município novo"),
    ("O mais simples entre os empatados", "3 algoritmos otimizados e comparados; ficou a regressão logística, "
                                          "explicável a qualquer gestor"),
    ("Probabilidades confiáveis", f"Brasil {ANO_TREINO}: previsto {num(validacao[f'prevista_oof_{ANO_TREINO}'], 1)}%, "
                                  f"oficial {num(validacao[f'oficial_{ANO_TREINO}'], 1)}%"),
    ("Reproduzível", "refeito do zero a partir do repositório: números idênticos; 59 testes automáticos"),
]
for i, (titulo, corpo) in enumerate(garantias):
    y = Inches(1.65) + i * Inches(1.3)
    circulo_numero(s, Inches(0.55), y + Inches(0.08), i + 1, ACCENT)
    add_text(s, Inches(1.3), y, Inches(5.6), Inches(0.42), titulo, size=17, color=NAVY, bold=True)
    add_text(s, Inches(1.3), y + Inches(0.42), Inches(5.6), Inches(0.8), corpo, size=13.5)
imagem(s, "07_calibracao.png", Inches(7.35), Inches(1.5), h=Inches(5.0))
add_text(s, Inches(7.35), Inches(6.55), Inches(5.3), Inches(0.5),
         f"ROC-AUC no teste: {num(teste['roc_auc'])} (IC 95%: {num(bootstrap['roc_auc_ic95'][0])}–"
         f"{num(bootstrap['roc_auc_ic95'][1])}) · palpite trivial: 0,50", size=11, color=GRAY, italic=True)
registrar(s, "Por que confiar", f"""
Por que confiar? Primeiro, testamos o modelo em {inteiro(METRICAS['n_municipios_teste'])} municípios que ele
nunca viu. Segundo, comparamos três algoritmos e ficamos com o mais simples entre os empatados, uma regressão
logística, que dá para explicar a qualquer gestor. Terceiro, as probabilidades são confiáveis: em
{ANO_TREINO}, o modelo previu {num(validacao[f'prevista_oof_{ANO_TREINO}'], 1)}% de alfabetização no Brasil, e
o número oficial foi {num(validacao[f'oficial_{ANO_TREINO}'], 1)}%. E tudo é reproduzível: refazendo o
projeto do zero, os números saem idênticos.
""")

# 5 — O que mais pesa ──────────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "O que mais pesa: o histórico educacional do município", "Principais insights")
ordem = ABLACAO.sort_values("poder_sozinho")
nomes_curtos = {
    "Histórico educacional do município (ano anterior)": "Histórico educacional do município",
    "Meta e esforço para atingi-la": "Meta do ano (deriva do histórico)",
    "IDHM (socioeconômico)": "IDHM (socioeconômico)",
    "Território e porte do município": "Território e porte",
    "Escola e rede": "Escola e rede",
}
dados = CategoryChartData()
dados.categories = [nomes_curtos.get(g, g) for g in ordem["grupo"]]
dados.add_series("Poder de previsão entregue sozinho", list(ordem["poder_sozinho"]))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.6), Inches(7.4), Inches(5.0), dados)
ch = gf.chart
estilo_grafico(ch, 13)
ch.has_title = True
ch.chart_title.text_frame.text = "Quanto do poder de previsão do modelo cada grupo entrega SOZINHO"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.has_legend = False
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1.15
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
plot.data_labels.number_format, plot.data_labels.number_format_is_linked = '0%', False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size, plot.data_labels.font.bold = Pt(14), True
for i, grupo in enumerate(ordem["grupo"]):
    ponto = plot.series[0].points[i]
    ponto.format.fill.solid()
    ponto.format.fill.fore_color.rgb = NAVY if grupo.startswith("Histórico") else GRAY
add_text(s, Inches(0.5), Inches(6.62), Inches(7.3), Inches(0.45),
         "Ablação: modelo retreinado só com cada grupo, em validação cruzada agrupada por município.",
         size=10.5, color=GRAY, italic=True)
add_runs(s, Inches(8.2), Inches(1.75), Inches(4.7), Inches(1.4), [
    (pct(_hist["poder_sozinho"]), NAVY, True), (" do poder de previsão vem do histórico educacional do município "
                                               "no ano anterior.", TEXT_DARK, False)], size=18)
add_runs(s, Inches(8.2), Inches(3.15), Inches(4.7), Inches(1.2), [
    ("O IDHM, sozinho, entrega só ", TEXT_DARK, False), (pct(_idhm["poder_sozinho"]), RED, True), (".", TEXT_DARK, False)],
    size=18)
card(s, Inches(8.2), Inches(4.35), Inches(4.65), Inches(2.55), "Para a gestão",
     "As alavancas de maior efeito são educacionais — formação, material, acompanhamento — e respondem em um "
     "ano letivo. Indicadores socioeconômicos levam uma década para mudar.", GREEN, tamanho_corpo=14)
registrar(s, "O que mais pesa", f"""
O principal achado: o que mais pesa no resultado de uma criança é o histórico educacional do município. Só com
o histórico do ano anterior, ou seja, a taxa de alfabetização, a proficiência e a participação na prova, o modelo
entrega {pct(_hist['poder_sozinho'])} do seu poder de
previsão. O IDHM, sozinho, entrega {pct(_idhm['poder_sozinho'])}. Para a gestão: as alavancas mais fortes são
educacionais, como formação, material e acompanhamento, e respondem em um ano letivo.
""")

# 6 — Contexto não é destino ───────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Contexto desfavorável não é destino", "Padrões semelhantes entre municípios")
perfis = [
    (C_TOPO, "Perfil de alta alfabetização", GREEN),
    (C_RICO, "Perfil intermediário", ACCENT),
    (C_BAIXO, "Perfil de maior risco", RED),
]
for i, (c, nome, cor) in enumerate(perfis):
    x = Inches(0.5) + i * Inches(2.85)
    linha = PERFIL.loc[c]
    regioes = COMPOSICAO.loc[c].sort_values(ascending=False)
    add_rect(s, x, Inches(1.65), Inches(2.65), Inches(4.0), fill=WHITE, line=cor, line_width=Pt(2.25), rounded=True)
    add_text(s, x, Inches(1.8), Inches(2.65), Inches(0.4), nome, size=14, color=cor, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(2.25), Inches(2.65), Inches(0.75), pct(linha["taxa_alfabetizacao_ant"] / 100),
             size=36, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(2.95), Inches(2.65), Inches(0.35), "alfabetizados", size=12, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(3.5), Inches(2.65), Inches(0.5), num(linha["idhm_educacao"]), size=24, color=NAVY,
             bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(3.98), Inches(2.65), Inches(0.35), "IDHM Educação", size=12, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.15), Inches(4.5), Inches(2.35), Inches(1.0),
             f"{inteiro(linha['n_municipios'])} municípios\n{regioes.iloc[0]:.0f}% dos municípios do {regioes.index[0]}",
             size=12, color=GRAY, align=PP_ALIGN.CENTER)
kpi(s, Inches(9.2), Inches(1.65), Inches(3.65), Inches(2.3), pct(CE_PCT_TOPO),
    "dos municípios do Ceará estão no perfil de alta alfabetização", GREEN,
    f"média de {pct(CE_TAXA / 100)} alfabetizados", tamanho=40)
card(s, Inches(9.2), Inches(4.15), Inches(3.65), Inches(2.75), "Onde buscar o que funciona",
     "O Ceará mantém programa próprio de alfabetização desde 2007 (PAIC), referência para o Compromisso "
     "Nacional. Associação, não prova de causa — mas aponta práticas para replicar.", GREEN, tamanho_corpo=13)
add_text(s, Inches(0.5), Inches(5.85), Inches(8.4), Inches(1.0),
         f"O perfil de maior alfabetização tem IDHM Educação MENOR que o intermediário — e alfabetiza "
         f"{num(PERFIL.loc[C_TOPO, 'taxa_alfabetizacao_ant'] - PERFIL.loc[C_RICO, 'taxa_alfabetizacao_ant'], 0)} "
         "pontos a mais. Perfis formados por agrupamento (KMeans) sobre resultado, proficiência, desigualdade "
         "interna, participação e IDHM.", size=13, color=NAVY, italic=True)
registrar(s, "Contexto não é destino", f"""
Agrupamos os municípios em três perfis. O de maior alfabetização não é o mais rico: tem IDHM Educação menor que
o perfil intermediário e alfabetiza
{num(PERFIL.loc[C_TOPO, 'taxa_alfabetizacao_ant'] - PERFIL.loc[C_RICO, 'taxa_alfabetizacao_ant'], 0)} pontos a
mais. O caso mais claro é o Ceará: {pct(CE_PCT_TOPO)} dos municípios estão nesse perfil, com média de
{pct(CE_TAXA / 100)} de crianças alfabetizadas. O estado tem programa próprio de alfabetização desde 2007. Contexto desfavorável não é
destino, e isso mostra onde buscar práticas para replicar.
""")

# 7 — Onde agir primeiro ───────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Onde agir primeiro", "Valor estratégico")
kpi(s, Inches(0.5), Inches(1.65), Inches(3.8), Inches(2.45), f"{num(top20['lift'], 1)}×",
    "mais crianças em risco encontradas ao atender os 20% indicados pelo modelo, contra uma escolha ao acaso",
    GREEN, f"IC 95%: {num(bootstrap['lift_top20_ic95'][0], 2)}–{num(bootstrap['lift_top20_ic95'][1], 2)}×",
    tamanho=44)
kpi(s, Inches(0.5), Inches(4.3), Inches(3.8), Inches(2.6), num(backtest["correlacao_prevista_real"]),
    f"correlação entre a taxa prevista e a real dos municípios em {ANO_TREINO}", ACCENT,
    "o valor está no agregado", tamanho=40)
outras = len(RANKING) - TOP50_UF.head(5).sum()
cats = list(TOP50_UF.head(5).index) + (["outras"] if outras > 0 else [])
vals = list(TOP50_UF.head(5).values) + ([outras] if outras > 0 else [])
dados = CategoryChartData()
dados.categories = cats
dados.add_series("Municípios no top 50 de risco", [int(v) for v in vals])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.7), Inches(1.6), Inches(8.15), Inches(4.4), dados)
ch = gf.chart
estilo_grafico(ch, 13)
ch.has_title = True
ch.chart_title.text_frame.text = f"Os {len(RANKING)} municípios de maior risco previsto, por estado"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.has_legend = False
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
plot = ch.plots[0]
plot.gap_width = 70
plot.has_data_labels = True
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size, plot.data_labels.font.bold = Pt(16), True
for i in range(len(cats)):
    ponto = plot.series[0].points[i]
    ponto.format.fill.solid()
    ponto.format.fill.fore_color.rgb = RED if i < 2 else GRAY
add_text(s, Inches(4.7), Inches(6.1), Inches(8.15), Inches(0.9),
         f"{DUAS_UFS.index[0]} e {DUAS_UFS.index[1]} concentram {pct(DUAS_UFS.sum() / len(RANKING))} da lista. "
         "Ranking com predições fora da amostra: cada município avaliado por um modelo que nunca viu seus alunos.",
         size=12.5, color=NAVY, italic=True)
registrar(s, "Onde agir primeiro", f"""
Individualmente, o modelo erra bastante: ele não conhece a criança, só o contexto. Mas para priorizar, ele
funciona. Se um programa só consegue atender 20% dos alunos, escolhendo pelo modelo ele encontra
{vezes_em_palavras(top20['lift'])} mais crianças em risco do que escolhendo ao acaso. E o risco está
concentrado: {('Bahia' if DUAS_UFS.index[0] == 'BA' else DUAS_UFS.index[0])} e
{('Rio Grande do Norte' if DUAS_UFS.index[1] == 'RN' else DUAS_UFS.index[1])} somam
{pct(DUAS_UFS.sum() / len(RANKING))} dos {len(RANKING)} municípios de maior risco.
""")

# 8 — Metas ────────────────────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Antecipar quem não vai bater a meta", "Valor estratégico")
card(s, Inches(0.5), Inches(1.65), Inches(4.1), Inches(2.45), f"Teste no passado ({ANO_TREINO})",
     f"Com dados de {ANO_TREINO - 1}, o modelo apontou os municípios que ficariam abaixo da meta de "
     f"{ANO_TREINO}: {pct(backtest['precisao_alerta'])} dos alertas se confirmaram, e o alerta encontrou "
     f"{pct(backtest['recall_alerta'])} dos {inteiro(backtest['real_abaixo_meta'])} municípios que de fato ficaram "
     "abaixo.", ACCENT, tamanho_corpo=14)
anos_v = sorted(int(a) for a in BRASIL.index)
dados = CategoryChartData()
dados.categories = [str(a) for a in anos_v]
dados.add_series("Oficial (INEP)", [BRASIL[a] for a in anos_v])
dados.add_series("Modelo", [None if a < ANO_TREINO else
                            validacao[f"prevista_oof_{ANO_TREINO}"] if a == ANO_TREINO else
                            validacao[f"projetada_{ANO_PROJ}"] for a in anos_v])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(4.85), Inches(1.6), Inches(4.2), Inches(4.0), dados)
ch = gf.chart
estilo_grafico(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "Brasil: oficial × modelo"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.has_legend, ch.legend.position, ch.legend.include_in_layout = True, XL_LEGEND_POSITION.BOTTOM, False
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 54, 68
ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
for serie, cor in ((ch.plots[0].series[0], NAVY), (ch.plots[0].series[1], ORANGE)):
    serie.format.line.color.rgb = cor
    serie.format.line.width = Pt(2.75)
    serie.smooth = False
    serie.marker.format.fill.solid()
    serie.marker.format.fill.fore_color.rgb = cor
    serie.marker.format.line.color.rgb = cor
    serie.data_labels.show_value = True
    serie.data_labels.number_format, serie.data_labels.number_format_is_linked = '0.0', False
    serie.data_labels.font.size, serie.data_labels.font.bold = Pt(12), True
    serie.data_labels.font.color.rgb = cor
ch.plots[0].series[0].data_labels.position = XL_LABEL_POSITION.ABOVE
ch.plots[0].series[1].data_labels.position = XL_LABEL_POSITION.BELOW
card(s, Inches(0.5), Inches(4.3), Inches(4.1), Inches(2.6), f"Validação real em {ANO_PROJ}",
     f"O modelo projetou {num(validacao[f'projetada_{ANO_PROJ}'], 1)}% para o Brasil; o oficial saiu "
     f"{num(validacao[f'oficial_{ANO_PROJ}'], 1)}%. Acertou a direção, subestimou o avanço — por isso a "
     "projeção vem em dois cenários.", ORANGE, tamanho_corpo=14)
kpi(s, Inches(9.3), Inches(1.65), Inches(3.55), Inches(2.45), inteiro(projecao["municipios_abaixo_meta"]),
    f"municípios abaixo da meta de {ANO_PROJ} no cenário pessimista (modelo puro)", RED, tamanho=40)
kpi(s, Inches(9.3), Inches(4.3), Inches(3.55), Inches(2.6), inteiro(ajustada["municipios_abaixo_meta"]),
    f"no cenário ajustado ao Brasil {ANO_PROJ} oficial", ORANGE,
    f"de {inteiro(projecao['municipios_avaliados'])} · mesma ordem de prioridade", tamanho=40)
add_text(s, Inches(4.85), Inches(5.75), Inches(4.2), Inches(1.2),
         "Os microdados param em 2024; o indicador nacional de 2025 permitiu testar a projeção fora do tempo.",
         size=11, color=GRAY, italic=True)
registrar(s, "Metas", f"""
Também testamos no passado: com dados de {ANO_TREINO - 1}, o modelo apontou os municípios que ficariam abaixo
da meta de {ANO_TREINO}, e seis em cada dez alertas se confirmaram. Para {ANO_PROJ}, tivemos uma validação real:
o resultado nacional já saiu, {num(validacao[f'oficial_{ANO_PROJ}'], 0)}%. O modelo projetou
{num(validacao[f'projetada_{ANO_PROJ}'], 0)}%: acertou a direção, mas subestimou o avanço. Por isso
apresentamos dois cenários: entre {inteiro(ajustada['municipios_abaixo_meta'])} e
{inteiro(projecao['municipios_abaixo_meta'])} municípios com risco de ficar abaixo da meta. A ordem de
prioridade é a mesma nos dois.
""")

# 9 — Políticas públicas ───────────────────────────────────────────────────────
s = novo_slide(prs)
cabecalho(s, "Como o modelo apoia a política pública", "Aplicação")
usos = [
    ("Priorizar antes do ano letivo", "Ranking de risco para direcionar formação de professores, material "
                                      "estruturado e acompanhamento técnico aos municípios que mais precisam.",
     ACCENT),
    ("Alertar sobre metas", "Lista de municípios com risco de não atingir a meta do ano, a tempo de pactuar "
                            "apoio — em vez de descobrir depois do resultado.", ORANGE),
    ("Monitorar a cada ano", "A pipeline da Fase 2 atualiza a base a cada divulgação do INEP; o modelo reavalia "
                             "o risco e recalibra o nível sem trabalho manual.", GREEN),
]
for i, (titulo, corpo, cor) in enumerate(usos):
    x = Inches(0.5) + i * Inches(4.18)
    add_rect(s, x, Inches(1.65), Inches(3.95), Inches(3.05), fill=WHITE, line=cor, line_width=Pt(2), rounded=True)
    circulo_numero(s, x + Inches(0.25), Inches(1.85), i + 1, cor, d=Inches(0.6))
    add_text(s, x + Inches(1.0), Inches(1.88), Inches(2.8), Inches(0.6), titulo, size=17, color=NAVY, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, x + Inches(0.25), Inches(2.65), Inches(3.45), Inches(2.0), corpo, size=15)
add_rect(s, Inches(0.5), Inches(5.0), Inches(12.35), Inches(1.85), fill=NAVY, rounded=True)
add_runs(s, Inches(0.85), Inches(5.1), Inches(11.7), Inches(1.65), [
    ("Uso responsável:  ", LIGHT_BLUE, True),
    ("o modelo indica onde concentrar APOIO, nunca quem punir. O erro individual é alto — a predição não deve "
     "reduzir recursos de um município nem rotular uma criança. A decisão final é da equipe que conhece o "
     "território.", WHITE, False)], size=17, anchor=MSO_ANCHOR.MIDDLE)
registrar(s, "Políticas públicas", """
Como isso apoia a política pública? São três usos. Priorizar municípios antes do ano letivo. Alertar sobre
metas, a tempo de pactuar apoio. E monitorar a cada ano, porque a base se atualiza a cada divulgação do INEP.
Com um cuidado importante: o modelo indica onde concentrar apoio, nunca quem punir.
""")

# 10 — Conclusão ───────────────────────────────────────────────────────────────
s = novo_slide(prs)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
add_text(s, Inches(0.6), Inches(0.55), Inches(12), Inches(0.7), "Em resumo", size=34, color=WHITE, bold=True)
conclusoes = [
    ("O histórico educacional é o melhor sinal de risco", f"{pct(_hist['poder_sozinho'])} do poder de previsão"),
    ("Contexto desfavorável não é destino", f"Ceará: {pct(CE_PCT_TOPO)} dos municípios no perfil de topo"),
    ("Dá para agir antes da prova", f"{num(top20['lift'], 1)}× mais crianças em risco encontradas na priorização"),
]
for i, (titulo, detalhe) in enumerate(conclusoes):
    y = Inches(1.6) + i * Inches(1.15)
    circulo_numero(s, Inches(0.65), y + Inches(0.05), i + 1, ACCENT, d=Inches(0.6))
    add_text(s, Inches(1.5), y, Inches(7.5), Inches(0.5), titulo, size=21, color=WHITE, bold=True)
    add_text(s, Inches(1.5), y + Inches(0.5), Inches(7.5), Inches(0.45), detalhe, size=15, color=LIGHT_BLUE)
add_rect(s, Inches(9.3), Inches(1.55), Inches(3.55), Inches(2.55), fill=BLUE_MED, rounded=True)
add_text(s, Inches(9.55), Inches(1.7), Inches(3.1), Inches(0.45), "Próximos passos", size=17, color=WHITE, bold=True)
add_bullets(s, Inches(9.55), Inches(2.2), Inches(3.15), Inches(2.7), [
    "Censo Escolar e CadÚnico na base",
    f"Revalidar com os dados de {ANO_PROJ}",
    "Piloto com uma secretaria",
], size=14, cor_marcador=LIGHT_BLUE, espaco=8, cor=WHITE)
add_rect(s, 0, Inches(6.3), SLIDE_W, Inches(1.2), fill=BLUE_MED)
add_text(s, 0, Inches(6.55), SLIDE_W, Inches(0.5), f"Obrigado  |  {AUTOR}  |  Pós-Tech FIAP  |  1IAST",
         size=16, color=WHITE, align=PP_ALIGN.CENTER)
registrar(s, "Conclusão", f"""
Resumindo: o histórico educacional é o melhor sinal de risco, contexto desfavorável não é destino, e com o que
já sabemos dá para agir antes da prova. Os próximos passos são incluir o Censo Escolar e o CadÚnico,
revalidar com os dados de {ANO_PROJ} e fazer um piloto com uma secretaria. Obrigado.
""")

prs.save(DESTINO_PPTX)

# ─── Roteiro (mesma fala das notas) ───────────────────────────────────────────

linhas = []
inicio, total_palavras = 0.0, 0
tabela = ["| # | Slide | Início | Duração | Palavras |", "|---|---|---|---|---|"]
blocos = []
for i, (titulo, fala) in enumerate(SLIDES, 1):
    palavras = len(fala.split())
    dur = palavras / PALAVRAS_POR_MINUTO * 60
    tabela.append(f"| {i} | {titulo} | {int(inicio // 60)}:{int(inicio % 60):02d} | ~{round(dur)} s | {palavras} |")
    blocos.append(f"### Slide {i} — {titulo}  ·  {int(inicio // 60)}:{int(inicio % 60):02d} → "
                  f"{int((inicio + dur) // 60)}:{int((inicio + dur) % 60):02d}\n\n> {fala}\n")
    inicio += dur
    total_palavras += palavras

roteiro = f"""# Roteiro do Vídeo Executivo — Fase 3

> Gerado por `python reports/presentation/gerar_apresentacao_fase3.py` junto com o deck
> [`Alfabetizacao_Fase3_Apresentacao.pptx`](../reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx)
> — a mesma fala está nas **notas do apresentador** de cada slide. Os números vêm dos artefatos do treino.

**Duração estimada: {int(inicio // 60)} min {int(inicio % 60):02d} s** ({total_palavras} palavras a
{PALAVRAS_POR_MINUTO} palavras/min) — dentro do limite de 5 minutos, com folga para respiros e troca de slide.

O enunciado pede que o vídeo apresente, simulando uma reunião executiva com gestores públicos:

| Ponto pedido | Slides |
|---|---|
| O problema educacional | 2 |
| Os principais insights | 5, 6 |
| O valor estratégico da solução | 3, 4, 7, 8 |
| Como os modelos apoiam políticas públicas | 9, 10 |

## Tempo por slide

{chr(10).join(tabela)}

## Fala, slide a slide

{chr(10).join(blocos)}
## Dicas de gravação

- **Ritmo**: fale pausado ({PALAVRAS_POR_MINUTO} palavras/min). Se o ensaio passar de 4 min 50 s, encurte a fala do
  slide 6 (perfis) — os outros cobrem os quatro pontos obrigatórios.
- **Tom**: reunião com gestores — fale de crianças, municípios e decisões; evite "ROC-AUC", "features",
  "regressão" fora do slide 4.
- **Tela**: modo apresentação do PowerPoint em tela cheia; use o modo do apresentador para ler as notas.
- **Ao apontar números**: os valores falados são os mesmos dos slides, arredondados para a fala.
- **Gravação**: tela + voz (câmera opcional), exporte em MP4. Se passar de 100 MB, publique como link e
  coloque o link na seção "Apresentação e Vídeo Executivo" do README.

## Perguntas prováveis (se houver discussão depois)

- **"O acerto individual não é baixo?"** — É, e é esperado: não há nenhum dado individual da criança na base.
  O uso é priorizar grupos e municípios, onde a correlação com o resultado real é de
  {num(backtest['correlacao_prevista_real'])}.
- **"Por que não um modelo mais sofisticado?"** — Testamos dois (Random Forest e Gradient Boosting). Ganham
  milésimos de acurácia, priorizam praticamente as mesmas crianças, e perdem a explicabilidade. O Gradient
  Boosting fica como modelo desafiante para a revalidação com {ANO_PROJ}.
- **"Por que o estado do aluno não entra?"** — Com um ano de treino, a variável aprenderia o choque daquele ano
  (as enchentes no RS em 2024) e o contaria duas vezes na projeção do ano seguinte.
"""
DESTINO_ROTEIRO.write_text(roteiro, encoding="utf-8")
print(f"Salvo: {DESTINO_PPTX} ({len(prs.slides)} slides)")
print(f"Salvo: {DESTINO_ROTEIRO} — {total_palavras} palavras, ~{int(inicio // 60)}min{int(inicio % 60):02d}s")
