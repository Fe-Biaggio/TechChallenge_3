# Model Card — Risco de Não Alfabetização (Tech Challenge Fase 3)

Ficha técnica do modelo no formato de *model card* (Mitchell et al., 2019):
o que ele faz, com que dados, quão bem, onde falha e como deve — e não
deve — ser usado. Detalhes e evidências de cada decisão em
[`decisoes_analiticas.md`](decisoes_analiticas.md).

## Identificação

| | |
|---|---|
| Artefato | [`models/modelo_final.joblib`](../models/modelo_final.joblib) — `sklearn.Pipeline` (pré-processamento + classificador) |
| Algoritmo | Regressão Logística com regularização L2 (`C = 0,01`) |
| Versão dos dados | Microdados INEP 2023–2024 + indicador nacional 2025, via pipeline da Fase 2 |
| Treino | `python -m src.modeling.treinar` (~18 min); reprodução verificada a partir de clone limpo, com números idênticos |
| Inferência | `python -m src.modeling.prever --ano-alvo 2025 [--taxa-nacional 66.0]` |

## Uso pretendido

- **Para que serve**: estimar, **antes da prova**, a probabilidade de cada
  aluno do 2º ano atingir o patamar de alfabetização (≥ 743 pontos Saeb) e,
  somando os alunos, a taxa esperada de cada município — para **priorizar
  apoio** (formação, material, acompanhamento) e **antecipar** municípios
  com risco de não atingir a meta.
- **Usuários**: gestores de secretarias estaduais/municipais de educação e
  equipes do Compromisso Nacional Criança Alfabetizada.
- **Fora do escopo**: decidir sobre um aluno individual; avaliar ou ranquear
  escolas/professores para fins de punição ou corte de recursos; inferir
  causalidade ("o que acontece se...").

## Dados

| | |
|---|---|
| População | 1.851.852 alunos de 2024 com prova válida (presentes e com caderno preenchido), 5.517 municípios |
| Alvo | `alfabetizado` — 59,8% positivos |
| Features (16) | Histórico educacional do município no ano anterior (6), meta do ano e esforço (2), IDHM 2010 — educação, renda, longevidade (3), escola e rede (2), território e porte (3) |
| Excluído por vazamento | proficiência (define o alvo), qualquer agregado do mesmo ano, presença, UF/região (choque estadual do próprio ano) — D1–D8 |

Nenhuma informação individual do aluno (frequência, histórico, família)
existe na base: o modelo vê o **contexto**, não a criança.

## Desempenho

Teste em 1.164 municípios **nunca vistos** (20% dos municípios, separados
antes de qualquer ajuste):

| Métrica | Modelo | Baseline (prevalência) |
|---|---|---|
| ROC-AUC | 0,628 (IC 95%: 0,594–0,655) | 0,500 |
| Brier score | 0,229 | 0,240 |
| Lift — 20% de maior risco | 1,45× (IC 95%: 1,36–1,49×) | 1,00× |
| ROC-AUC out-of-fold, todos os municípios | 0,645 | — |

No agregado (uso principal):

| | |
|---|---|
| Correlação taxa prevista × real por município (2024, out-of-fold) | 0,74 |
| Alertas de meta 2024 que se confirmaram | 61% |
| Brasil 2024: previsto × oficial | 59,3% × 59,2% |
| Brasil 2025: projetado × oficial (fora do tempo) | 61,1% × 66,0% |

## Desempenho por subgrupo

| Região | ROC-AUC | Taxa real | Taxa prevista |
|---|---|---|---|
| Nordeste | 0,668 | 54,0% | 55,1% |
| Sul | 0,614 | 63,1% | 66,3% |
| Centro-Oeste | 0,607 | 67,6% | 63,6% |
| Norte | 0,606 | 51,8% | 51,0% |
| Sudeste | 0,582 | 62,9% | 57,5% |

Rede estadual: ROC-AUC 0,569 e superestimada (64,3% previsto × 60,8% real);
rede municipal: 0,632. Rede privada: ausente da base (24 alunos).

## Limitações e riscos

1. **Erro individual alto** — ROC-AUC de 0,63 não sustenta decisões sobre um
   aluno. O valor está na priorização de grupos e no agregado municipal.
2. **Um único par de anos** (2023→2024) para aprender a relação; o modelo
   **não antecipa mudanças de ritmo nacional** — subestimou 2025 em 4,9 p.p.
   Mitigação: ajuste de nível ao resultado nacional mais recente (D20).
3. **Viés regional residual** — subestima o Sudeste e superestima o Sul e a
   rede estadual. Leituras regionais devem considerar o erro da tabela acima.
4. **IDHM de 2010** — envelhece mal onde o município mudou muito.
5. **Associação, não causa** — o ranking diz onde o risco está, não o que
   funciona para reduzi-lo.

## Uso responsável

- Usar para **direcionar apoio**, nunca para reduzir recursos ou rotular.
- Apresentar sempre o ranking com a incerteza (intervalo, cenário
  pessimista e ajustado), não como lista definitiva.
- Revalidar a cada nova divulgação do INEP; o HistGradientBoosting está
  registrado como **modelo desafiante** (D21) para essa revalidação.
- Decisão final com equipe técnica que conhece o território: o modelo é
  uma triagem, não um veredito.
