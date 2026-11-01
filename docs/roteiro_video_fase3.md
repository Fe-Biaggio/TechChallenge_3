# Roteiro do Vídeo Executivo — Fase 3

> Gerado por `python reports/presentation/gerar_apresentacao_fase3.py` junto com o deck
> [`Alfabetizacao_Fase3_Apresentacao.pptx`](../reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx)
> — a mesma fala está nas **notas do apresentador** de cada slide. Os números vêm dos artefatos do treino.

**Duração estimada: 4 min 34 s** (663 palavras a
145 palavras/min) — dentro do limite de 5 minutos, com folga para respiros e troca de slide.

O enunciado pede que o vídeo apresente, simulando uma reunião executiva com gestores públicos:

| Ponto pedido | Slides |
|---|---|
| O problema educacional | 2 |
| Os principais insights | 5, 6 |
| O valor estratégico da solução | 3, 4, 7, 8 |
| Como os modelos apoiam políticas públicas | 9, 10 |

## Tempo por slide

| # | Slide | Início | Duração | Palavras |
|---|---|---|---|---|
| 1 | Abertura | 0:00 | ~14 s | 35 |
| 2 | O problema educacional | 0:14 | ~38 s | 91 |
| 3 | A solução | 0:52 | ~33 s | 79 |
| 4 | Por que confiar | 1:24 | ~29 s | 71 |
| 5 | O que mais pesa | 1:54 | ~30 s | 72 |
| 6 | Contexto não é destino | 2:24 | ~29 s | 71 |
| 7 | Onde agir primeiro | 2:53 | ~26 s | 64 |
| 8 | Metas | 3:19 | ~32 s | 77 |
| 9 | Políticas públicas | 3:51 | ~21 s | 51 |
| 10 | Conclusão | 4:12 | ~22 s | 52 |

## Fala, slide a slide

### Slide 1 — Abertura  ·  0:00 → 0:14

> Olá. Eu sou Felipe Barbato de Biaggio, e este é o Case 3 do Tech Challenge. Vou mostrar como antecipar, antes da prova, quais crianças e quais municípios estão em risco de não se alfabetizar.

### Slide 2 — O problema educacional  ·  0:14 → 0:52

> O Brasil se comprometeu a alfabetizar todas as crianças até o fim do segundo ano, até 2030. O indicador nacional avançou: de 56% em 2023 para 66% em 2025. Mesmo assim, uma em cada três crianças ainda termina o segundo ano sem estar alfabetizada, e quase metade dos municípios ficou abaixo da própria meta em 2024. O problema é de tempo: quando o resultado sai, o ano letivo já acabou. A nossa pergunta foi: com o que já se sabe no início do ano, dá para antecipar onde está o risco?

### Slide 3 — A solução  ·  0:52 → 1:24

> Usamos a base construída na fase anterior, com 1,85 milhão de alunos avaliados em 2024, e treinamos um modelo que estima, para cada aluno, a chance de se alfabetizar. Somando os alunos de um município, temos a taxa esperada, que comparamos com a meta. A regra de ouro: o modelo só usa o que existe antes da prova, como o desempenho do município no ano anterior, a meta, a escola e o IDHM. Nada do resultado que queremos prever.

### Slide 4 — Por que confiar  ·  1:24 → 1:54

> Por que confiar? Primeiro, testamos o modelo em 1.164 municípios que ele nunca viu. Segundo, comparamos três algoritmos e ficamos com o mais simples entre os empatados, uma regressão logística, que dá para explicar a qualquer gestor. Terceiro, as probabilidades são confiáveis: em 2024, o modelo previu 59,3% de alfabetização no Brasil, e o número oficial foi 59,2%. E tudo é reproduzível: refazendo o projeto do zero, os números saem idênticos.

### Slide 5 — O que mais pesa  ·  1:54 → 2:24

> O principal achado: o que mais pesa no resultado de uma criança é o histórico educacional do município. Só com o histórico do ano anterior, ou seja, a taxa de alfabetização, a proficiência e a participação na prova, o modelo entrega 97% do seu poder de previsão. O IDHM, sozinho, entrega 26%. Para a gestão: as alavancas mais fortes são educacionais, como formação, material e acompanhamento, e respondem em um ano letivo.

### Slide 6 — Contexto não é destino  ·  2:24 → 2:53

> Agrupamos os municípios em três perfis. O de maior alfabetização não é o mais rico: tem IDHM Educação menor que o perfil intermediário e alfabetiza 21 pontos a mais. O caso mais claro é o Ceará: 93% dos municípios estão nesse perfil, com média de 90% de crianças alfabetizadas. O estado tem programa próprio de alfabetização desde 2007. Contexto desfavorável não é destino, e isso mostra onde buscar práticas para replicar.

### Slide 7 — Onde agir primeiro  ·  2:53 → 3:19

> Individualmente, o modelo erra bastante: ele não conhece a criança, só o contexto. Mas para priorizar, ele funciona. Se um programa só consegue atender 20% dos alunos, escolhendo pelo modelo ele encontra uma vez e meia mais crianças em risco do que escolhendo ao acaso. E o risco está concentrado: Bahia e Rio Grande do Norte somam 66% dos 50 municípios de maior risco.

### Slide 8 — Metas  ·  3:19 → 3:51

> Também testamos no passado: com dados de 2023, o modelo apontou os municípios que ficariam abaixo da meta de 2024, e seis em cada dez alertas se confirmaram. Para 2025, tivemos uma validação real: o resultado nacional já saiu, 66%. O modelo projetou 61%: acertou a direção, mas subestimou o avanço. Por isso apresentamos dois cenários: entre 1.757 e 2.382 municípios com risco de ficar abaixo da meta. A ordem de prioridade é a mesma nos dois.

### Slide 9 — Políticas públicas  ·  3:51 → 4:12

> Como isso apoia a política pública? São três usos. Priorizar municípios antes do ano letivo. Alertar sobre metas, a tempo de pactuar apoio. E monitorar a cada ano, porque a base se atualiza a cada divulgação do INEP. Com um cuidado importante: o modelo indica onde concentrar apoio, nunca quem punir.

### Slide 10 — Conclusão  ·  4:12 → 4:34

> Resumindo: o histórico educacional é o melhor sinal de risco, contexto desfavorável não é destino, e com o que já sabemos dá para agir antes da prova. Os próximos passos são incluir o Censo Escolar e o CadÚnico, revalidar com os dados de 2025 e fazer um piloto com uma secretaria. Obrigado.

## Dicas de gravação

- **Ritmo**: fale pausado (145 palavras/min). Se o ensaio passar de 4 min 50 s, encurte a fala do
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
  0,74.
- **"Por que não um modelo mais sofisticado?"** — Testamos dois (Random Forest e Gradient Boosting). Ganham
  milésimos de acurácia, priorizam praticamente as mesmas crianças, e perdem a explicabilidade. O Gradient
  Boosting fica como modelo desafiante para a revalidação com 2025.
- **"Por que o estado do aluno não entra?"** — Com um ano de treino, a variável aprenderia o choque daquele ano
  (as enchentes no RS em 2024) e o contaria duas vezes na projeção do ano seguinte.
