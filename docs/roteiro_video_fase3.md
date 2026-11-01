# Roteiro do Vídeo Executivo — Fase 3

> Gerado por `python reports/presentation/gerar_apresentacao_fase3.py` junto com o deck
> [`Alfabetizacao_Fase3_Apresentacao.pptx`](../reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx)
> — a mesma fala está nas **notas do apresentador** de cada slide. Os números vêm dos artefatos do treino.

**Duração estimada: 4 min 21 s** (544 palavras, lidas a
135 palavras/min, com cada número contando como 3 palavras faladas) — dentro do
limite de 5 minutos, com folga para respiros e troca de slide.

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
| 1 | Abertura | 0:00 | ~13 s | 28 |
| 2 | O problema educacional | 0:13 | ~37 s | 76 |
| 3 | A solução | 0:50 | ~27 s | 60 |
| 4 | Por que confiar | 1:17 | ~26 s | 52 |
| 5 | O que mais pesa | 1:43 | ~28 s | 59 |
| 6 | Contexto não é destino | 2:11 | ~28 s | 56 |
| 7 | Onde agir primeiro | 2:38 | ~30 s | 61 |
| 8 | Metas | 3:08 | ~32 s | 61 |
| 9 | Políticas públicas | 3:40 | ~21 s | 47 |
| 10 | Conclusão | 4:00 | ~20 s | 44 |

## Fala, slide a slide

### Slide 1 — Abertura  ·  0:00 → 0:13

> Olá, eu sou Felipe Barbato de Biaggio. Neste Case 3, mostro como antecipar, antes da prova, quais crianças e quais municípios estão em risco de não se alfabetizar.

### Slide 2 — O problema educacional  ·  0:13 → 0:50

> O Brasil se comprometeu a alfabetizar todas as crianças até o fim do segundo ano, até 2030. O país avançou, de 56 para 66% em dois anos. Mas uma em cada três crianças ainda termina o segundo ano sem estar alfabetizada, e quase metade dos municípios ficou abaixo da própria meta em 2024. E o resultado chega tarde: quando sai, o ano letivo já acabou. A pergunta é: dá para antecipar o risco antes da prova?

### Slide 3 — A solução  ·  0:50 → 1:17

> Com a base da fase anterior, quase dois milhões de alunos, treinamos um modelo que estima a chance de cada criança se alfabetizar. Somando por município, temos a taxa esperada, comparável com a meta. A regra de ouro: o modelo só usa o que existe antes da prova, como o histórico do município, a meta, a escola e o IDHM.

### Slide 4 — Por que confiar  ·  1:17 → 1:43

> Por que confiar? Testamos o modelo em mais de mil municípios que ele nunca viu. Entre três algoritmos, ficamos com o mais simples entre os empatados, uma regressão logística, explicável a qualquer gestor. E as probabilidades são confiáveis: para o Brasil em 2024, o modelo previu 59,3%, e o oficial foi 59,2%.

### Slide 5 — O que mais pesa  ·  1:43 → 2:11

> O principal achado: o que mais pesa é o histórico educacional do município. Só com a taxa, a proficiência e a participação do ano anterior, o modelo entrega 97% do seu poder de previsão; o IDHM, sozinho, só 26%. Para a gestão: as alavancas mais fortes são educacionais, como formação, material e acompanhamento, e respondem em um ano letivo.

### Slide 6 — Contexto não é destino  ·  2:11 → 2:38

> E contexto não é destino. O perfil de municípios com maior alfabetização tem IDHM menor que o intermediário, e alfabetiza 21 pontos a mais. O Ceará é o caso mais claro: 93% dos seus municípios estão nesse perfil, e o estado tem programa próprio de alfabetização desde 2007. É ali que vale buscar práticas para replicar.

### Slide 7 — Onde agir primeiro  ·  2:38 → 3:08

> Para uma criança específica, o modelo erra bastante: ele vê o contexto, não a criança. Mas para priorizar, ele funciona. Atendendo os 20% indicados pelo modelo, um programa encontra uma vez e meia mais crianças em risco do que escolhendo ao acaso. E o risco está concentrado: Bahia e Rio Grande do Norte somam 66% dos 50 municípios de maior risco.

### Slide 8 — Metas  ·  3:08 → 3:40

> No teste com o passado, seis em cada dez alertas de meta se confirmaram. Para 2025, tivemos uma validação real: o Brasil fez 66%, e o modelo projetou 61%. Acertou a direção, mas subestimou o avanço. Por isso trabalhamos com dois cenários: entre 37% e 50% dos municípios com risco de ficar abaixo da meta, com a mesma ordem de prioridade.

### Slide 9 — Políticas públicas  ·  3:40 → 4:00

> Na prática, são três usos: priorizar municípios antes do ano letivo, alertar sobre metas a tempo de pactuar apoio e monitorar a cada ano, já que a base se atualiza a cada divulgação do INEP. Com um cuidado: o modelo indica onde concentrar apoio, nunca quem punir.

### Slide 10 — Conclusão  ·  4:00 → 4:21

> Resumindo: o histórico educacional é o melhor sinal de risco, contexto não é destino, e dá para agir antes da prova. Próximos passos: incluir o Censo Escolar e o CadÚnico, revalidar com os dados de 2025 e fazer um piloto com uma secretaria. Obrigado.

## Dicas de gravação

- **Ritmo**: fale pausado (~135 palavras/min). Cronometre um ensaio: se passar de 4 min 50 s,
  encurte a fala do slide 6 (perfis) — os outros cobrem os quatro pontos obrigatórios.
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
