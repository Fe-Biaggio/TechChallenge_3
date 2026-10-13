# Roteiro — Vídeo Executivo da Fase 3 (até 5 minutos)

Apoio visual: [`reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx`](../reports/presentation/Alfabetizacao_Fase3_Apresentacao.pptx)
(gerado por `python reports/presentation/gerar_apresentacao_fase3.py`).

O enunciado pede quatro pontos — cada bloco abaixo indica qual cobre:

| Ponto pedido | Blocos |
|---|---|
| Problema educacional | 1 |
| Principais insights | 3, 4 |
| Valor estratégico da solução | 5, 6 |
| Como os modelos apoiam políticas públicas | 7 |

Tom: apresentação para a liderança de uma secretaria de educação — sem
jargão técnico; o rigor aparece como "por que confiar", não como métrica.
Tempo-alvo por bloco entre colchetes; total ≈ 4min40s (~640 palavras a
~140 palavras/min).

---

### Abertura — slide 1 [0:00–0:15]

> Quem está em risco de não se alfabetizar? Neste projeto, construímos um
> modelo que responde essa pergunta **antes** da prova — a tempo de agir.

### 1. O problema educacional — slide 2 [0:15–0:50]

> O Brasil se comprometeu a alfabetizar todas as crianças até o fim do 2º
> ano do fundamental até 2030. O indicador nacional vem subindo — de 56% em
> 2023 para 66% em 2025 —, mas um terço das crianças ainda termina o 2º ano
> sem estar alfabetizada, e quase metade dos municípios ficou abaixo da
> própria meta em 2024.
>
> O problema é de tempo: quando o resultado sai, o ano letivo já acabou. A
> pergunta desta fase foi: com o que já se sabe no início do ano, dá para
> antecipar quais alunos e quais municípios estão em risco?

### 2. A solução — slides 3 e 4 [0:50–1:40]

> Usamos a base que construímos na fase anterior — 1,85 milhão de alunos
> avaliados em 2024 — e treinamos um modelo que estima, para cada aluno, a
> chance de atingir o patamar de alfabetização. Somando os alunos de um
> município, temos a taxa esperada, que pode ser comparada com a meta.
>
> A regra mais importante: o modelo só usa o que existe **antes** da prova
> — o desempenho do município no ano anterior, a meta, o porte da escola, a
> rede e indicadores de desenvolvimento humano. Nada do resultado que
> estamos tentando prever.
>
> Por que confiar: testamos em mais de mil municípios que o modelo nunca
> viu, comparamos com um palpite trivial, e escolhemos o modelo mais simples
> entre os que empataram — uma regressão logística, fácil de explicar. E as
> probabilidades são calibradas: quando o modelo diz 60%, cerca de 60% de
> fato se alfabetizam.

### 3. Principais insights — slide 5 [1:40–2:20]

> O que mais pesa no resultado de um aluno é o **histórico educacional do
> município**: a taxa de alfabetização e a proficiência do ano anterior.
> Indicadores socioeconômicos amplos, como o IDHM, pesam bem menos.
>
> A mensagem para a gestão: as alavancas de maior efeito são de gestão
> educacional — formação, material, acompanhamento —, que respondem em um
> ano letivo, e não só fatores estruturais, que levam uma década para mudar.

### 4. Padrões regionais — slide 8 [2:20–2:50]

> Agrupamos os municípios em três perfis. O de maior alfabetização **não**
> é o mais rico. O caso mais claro é o Ceará: mais de 90% dos municípios
> estão no perfil de topo, com IDHM Educação bem abaixo do perfil mais
> rico — que alfabetiza só 62% das crianças. O estado tem
> programa próprio de alfabetização há quase duas décadas. Contexto
> desfavorável não é destino — e isso indica onde buscar práticas para
> replicar.

### 5. Valor estratégico: onde agir primeiro — slide 6 [2:50–3:25]

> Individualmente, o modelo erra bastante — ele não conhece a criança, só o
> contexto. Mas para **priorizar**, funciona: se um programa só consegue
> atender 20% dos alunos, escolhendo pelo modelo ele encontra 1,5 vez mais
> crianças em risco do que escolhendo ao acaso. No nível do município, a
> taxa prevista tem correlação de 0,74 com a real.

### 6. Valor estratégico: metas — slide 7 [3:25–4:05]

> Fizemos o teste no passado: com dados de 2023, o modelo apontou os
> municípios que ficariam abaixo da meta de 2024 — e acertou 6 em cada 10.
>
> Para 2025, temos uma validação real: o resultado nacional já saiu, 66%. O
> modelo projetou 61% — acertou a direção, mas subestimou o avanço. Por
> isso apresentamos dois cenários: entre 1.757 e 2.382 municípios com risco
> de ficar abaixo da meta. A ordem de prioridade é a mesma nos dois.

### 7. Como apoia políticas públicas — slides 9 e 10 [4:05–4:40]

> Três usos concretos: priorizar municípios antes do ano letivo; alertar
> sobre metas a tempo de pactuar apoio; e monitorar a cada ano, porque a
> base se atualiza a cada divulgação do INEP.
>
> Um cuidado: o modelo indica onde concentrar **apoio**, nunca quem punir.
> Os próximos passos são incluir dados do Censo Escolar e do CadÚnico e
> fazer um piloto com uma secretaria para medir o efeito da priorização.
>
> Obrigado.

---

## Checklist de gravação

- [ ] Gerar o deck atualizado (`python reports/presentation/gerar_apresentacao_fase3.py`)
- [ ] Ensaiar com cronômetro — cortar o bloco 4 se passar de 4min50s
- [ ] Gravar tela (slides) + voz; câmera opcional
- [ ] Exportar em MP4, salvar em `reports/presentation/` (ou link no README se > 100 MB)
- [ ] Atualizar o link na seção "Apresentação e Vídeo Executivo" do README
