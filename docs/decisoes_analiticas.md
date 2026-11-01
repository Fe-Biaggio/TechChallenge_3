# Decisões Analíticas — Fase 3

Registro das decisões de modelagem que mudam o resultado, cada uma com o
contexto, a evidência nos dados e as alternativas descartadas. Pensado para
quem vai revisar ou continuar o projeto: o "porquê" de cada escolha está
aqui, não espalhado em comentários.

Formato: **D#** — decisão · *Evidência* · *Alternativas descartadas*.

---

## Fronteira de vazamento

### D1 — Princípio único: toda feature precisa existir antes da prova do ano previsto

Todas as decisões de inclusão/exclusão abaixo derivam desta regra. Ela
torna a fronteira de vazamento verificável (existe teste automatizado que
falha se uma feature dinâmica do mesmo ano entrar — ver
`tests/test_ml_pipeline.py::test_todo_contexto_municipal_e_do_ano_anterior`)
e corresponde ao uso real: um sistema de alerta precoce prevê **antes** da
avaliação, para ainda dar tempo de agir.

### D2 — `proficiencia` excluída

`alfabetizado` é definido por corte determinístico sobre `proficiencia`
(≥ 743 pontos na escala Saeb).

*Evidência*: nos dados, o maior valor de `proficiencia` entre alunos
`alfabetizado=False` é 742,999; o menor entre `alfabetizado=True` está logo
acima de 743. Usá-la seria prever o rótulo a partir de si mesmo.

### D3 — Contexto municipal sempre do ano anterior (Y−1)

O indicador municipal do próprio ano é um agregado que **inclui o aluno que
está sendo previsto** — em municípios pequenos, o resultado de um aluno move
o agregado de forma perceptível (vazamento por agregação circular). Todo o
contexto municipal entra defasado: resultado oficial, resultado da rede
municipal, nível, participação e proficiência média/dispersão de Y−1.

*Alternativa descartada*: usar o indicador do mesmo ano "porque é público".
Ele é público só **depois** da prova — viola D1.

### D4 — Meta do ano Y entra, mas só a versão publicada em Y−1

A tabela de metas tem "vintages": a trajetória publicada em 2023 é revisada
em 2024. A meta de 2024 usada como feature vem da vintage de 2023 (lida da
Bronze, não da Silver — a Silver guarda só a vintage mais recente, que para
este fim seria a revisão feita já no ano da prova).

`esforco_meta = meta(Y) − resultado da rede municipal(Y−1)` — quanto o
município precisaria avançar. É a feature numérica mais correlacionada
(negativamente) com o alvo entre as que não são proficiência: −0,20.

### D5 — `presenca` e `preenchimento_caderno` definem a população, não são features

Só existem no dia da prova (violam D1). São usadas apenas para filtrar a
população: alunos que faltaram ou não preencheram o caderno recebem
`alfabetizado=False` por regra administrativa, não por medição — incluí-los
ensinaria o modelo a prever ausência.

*Evidência* (2023 + 2024): todos os 512 mil ausentes têm `proficiencia`
nula e `alfabetizado=False`; os 1.185 presentes com caderno não preenchido,
o mesmo. População final de treino: 1.851.852 alunos de 2024 (de 2.120.560
listados).

### D6 — `id_escola` não serve como chave entre anos

A ideia óbvia de feature engineering seria "taxa de alfabetização da
escola no ano anterior". Testada e rejeitada:

*Evidência*: dentro de um ano, cada código de escola pertence a um único
município (0 colisões em 2023 e em 2024). Mas entre anos, só **865 de
~37 mil** pares (município, escola) de 2023 reaparecem em 2024 — o código
é re-mascarado a cada edição (o dicionário da fonte o descreve como
"máscara do código da escola, códigos fictícios"). Um join por `id_escola`
ligaria escolas diferentes.

*O que ficou*: `escola_n_alunos` (tamanho da lista de alunos da escola na
avaliação daquele ano) — conhecido antes da prova e não depende do código
ser estável entre anos.

### D7 — Redundâncias removidas: `indicador_nacional`, `gap_indicador_nacional`, `faixa_risco`

Correção de um erro da primeira versão do modelo desta fase, que tinha
estas três como features.

*Evidência*: dentro de um único ano de treino, `indicador_nacional` é
constante (55,9 para todo aluno) e `gap_indicador_nacional =
taxa_municipal − 55,9` tem **correlação 1,0** com a taxa municipal. A
primeira versão apresentava as duas como "as duas features mais
importantes" — eram a mesma variável, e a importância estava sendo
dividida entre cópias. `faixa_risco` é só a discretização da mesma taxa.
Ficam para comunicar resultados, não como feature.

**Redundâncias por construção** — mesma lógica, achadas na revisão dos
coeficientes do modelo (estimados no treino, não no teste):

- `idhm` é a média geométrica de `idhm_educacao`, `idhm_longevidade` e
  `idhm_renda` (erro máximo de 0,0005 nos 5.565 municípios — só
  arredondamento). Com os quatro no modelo, a regressão logística dava
  coeficiente **−1,19** ao IDHM e +0,65/+0,45/+0,22 aos componentes: os
  termos se compensavam, e nenhum coeficiente podia ser lido isoladamente.
- `taxa_rede_municipal_ant` é exatamente `meta_ano − esforco_meta`. Continua
  sendo calculada (é a base de `esforco_meta`), mas não entra como feature.

Ficam os três componentes do IDHM e o par meta/esforço. Nenhuma informação é
perdida — as variáveis removidas são funções exatas das que ficaram. Há um
teste automatizado (`test_nenhuma_feature_e_combinacao_exata_de_outras`).

### D8 — UF e região fora das features (continuam nas análises)

Correção feita durante a revisão, depois de ver o modelo errar de forma
óbvia na projeção.

Com um único ano de treino, o coeficiente de cada estado aprende *o que
aconteceu com aquele estado naquele ano* — a partir dos resultados de
outros alunos do mesmo estado na mesma prova. É o mesmo vazamento de D3
(agregado do resultado do próprio ano), só que no nível estadual: o
"efeito RS em 2024" só é conhecido depois da prova de 2024, então viola D1.

*Evidência*: com UF como feature, a projeção para 2025 dava **27,7%** de
alfabetização para o RS (real de 2024: 51,6%) e marcava **264 de 264**
municípios gaúchos abaixo da meta. A queda de 2024 no RS (enchentes de
maio/2024) entrava duas vezes: no indicador defasado de 2024 (correto) e no
coeficiente do estado aprendido com o mesmo choque (dupla contagem). Na
avaliação no próprio ano, a UF parecia a 2ª feature mais importante —
justamente por capturar o choque daquele ano, que no teste (mesmo ano,
mesmo estado) também está presente.

*Mesma regra para as metas e indicadores estaduais*: o enunciado lista
"metas nacionais e estaduais" entre as integrações da Gold. A meta estadual
e o indicador da UF são **constantes para todos os municípios do estado** —
como feature, seriam um identificador da UF com outro nome e herdariam o
mesmo problema. A meta que se aplica ao aluno é a do **seu município**
(`meta_ano`, D4); a meta nacional entra na validação temporal (D20).

*Custo aceito*: alguma perda de ROC-AUC no teste do mesmo ano, em troca de
um modelo que transfere para o ano seguinte — que é o uso real. Diferenças
estruturais entre territórios continuam representadas pelo desempenho
defasado do município e pelo IDHM.

---

## População e período

### D9 — Treino só com 2024

Os microdados publicados cobrem 2023 e 2024. Alunos de 2023 não têm ano
anterior (não existe 2022) para alimentar o contexto de D3 — ficariam com
todas as features municipais nulas. Incluí-los ensinaria o modelo a ignorar
essas variáveis; usar 2023 como treino e 2024 como teste faria o modelo
treinar sem nunca ver as features preenchidas.

*Consequência aceita*: a relação "contexto de Y−1 → resultado de Y" é
aprendida a partir de um único par de anos (2023→2024). A projeção para
2025 assume que essa relação se mantém (ver limitações no README).

---

## Validação

### D10 — Teste por município, não por aluno

20% dos **municípios** (1.164) ficam fora de tudo — busca de
hiperparâmetros, comparação de modelos e treino. Split aleatório por aluno
colocaria colegas do mesmo município (mesmas features de contexto) no
treino e no teste, superestimando o desempenho. Testar em municípios nunca
vistos mede a pergunta real: o modelo funciona num município novo?

Toda validação cruzada (busca de hiperparâmetros, comparação de modelos,
predições out-of-fold) usa `StratifiedGroupKFold` agrupado por município,
pelo mesmo motivo. Existe uma asserção no código de treino que interrompe
a execução se algum município aparecer nos dois lados.

### D11 — Baseline trivial sempre reportado

Com 59,8% de alfabetizados, "prever sempre alfabetizado" acerta 59,8% —
acurácia sozinha engana. Todo resultado é reportado contra um
`DummyClassifier(strategy="prior")`, e a métrica de escolha é ROC-AUC
(independe de limiar e de prevalência).

### D12 — Sem `class_weight="balanced"`

A primeira versão usava. Com 60/40, o desbalanceamento é leve, e o
reponderamento distorce as probabilidades (puxa tudo para 0,5). Como as
probabilidades são agregadas por município para comparar com metas
(D16), precisam estar calibradas — verificado na curva de calibração
(`images/07_calibracao.png`) e no Brier score.

### D13 — Busca de hiperparâmetros + early stopping como estratégia contra overfitting

`RandomizedSearchCV` (15 configurações × 3 folds agrupados) sobre
taxa de aprendizado, número de folhas, mínimo de amostras por folha,
regularização L2 e profundidade do HistGradientBoosting, numa subamostra
de 300 mil alunos do treino. O próprio HGB usa early stopping (para quando
a perda numa fração interna de validação para de melhorar). O gap entre
ROC-AUC de treino e de validação é reportado para cada candidato.

*Por que subamostra*: a busca roda 45 treinos; em 1,48M alunos isso
levaria horas. O modelo escolhido é retreinado no treino completo.

### D14 — Seleção de modelo pela regra de 1 desvio padrão

Entre os candidatos cuja ROC-AUC média de validação fica a no máximo 1
desvio padrão do melhor, escolhe-se o **mais simples** (Regressão
Logística < Random Forest < HistGradientBoosting).

*Evidência que motivou*: numa primeira execução, o HGB "venceu" a
logística por 0,0009 de ROC-AUC, com desvio padrão entre folds de 0,0167
— diferença 18× menor que o ruído. O HGB tinha gap treino→validação de
0,027; a logística, de 0,004. Escolher o mais complexo por uma diferença
dentro do ruído é apostar em overfitting. A regra está em
`escolher_modelo()` (`src/modeling/treinar.py`), com testes para os dois
casos (empate → mais simples; diferença real → melhor).

### D15 — Uso operacional por priorização, não por limiar fixo

No limiar padrão (0,5), o modelo só identifica ~31% dos alunos em risco.
Maximizar F1 da classe de risco também foi testado e descartado: leva a um
limiar que marca quase todo mundo (recall 98%, precisão pouco acima da
prevalência) — infla a métrica sem ajudar ninguém.

O critério adotado reflete o uso real: um programa com capacidade para
atender X% dos alunos atende os X% de **maior risco previsto**. Reportamos,
para X de 5% a 50%, a taxa real de risco no grupo priorizado e o *lift*
sobre escolher ao acaso (`reports/analise_priorizacao.csv`).

---

## Aplicação estratégica

### D16 — Agregação municipal ponderada e restrita à rede Municipal para metas

Taxas municipais (real e prevista) usam `peso_aluno`, como o INEP no cálculo
oficial. A comparação com metas municipais usa só alunos da rede
Municipal — é a rede para a qual essas metas são definidas.

### D17 — Ranking e backtest com predições out-of-fold

Cada aluno é previsto por um modelo que nunca viu nenhum aluno do seu
município. Sem isso, o ranking de risco e o backtest de metas seriam
avaliados em dados de treino.

### D18 — Projeção 2025 usa a composição de alunos de 2024

Não existem microdados de 2025. A projeção mantém os alunos/escolas de
2024 (rede e porte) e troca o contexto municipal para o de 2024 e a meta
para a de 2025. Premissa explícita: a composição de rede e porte das
escolas muda pouco de um ano para o seguinte; o que muda é o contexto.

### D19 — Clusters por perfil municipal observado, não pela predição

Os grupos de municípios "com padrões semelhantes" são formados pelo perfil
observado (taxa, proficiência, desigualdade interna, participação, IDHM
educação e renda), não pela saída do modelo — para responder "quem se
parece com quem" sem circularidade. k entre 3 e 7, escolhido pelo maior
silhouette.

### D20 — Validação fora do tempo no nível nacional e projeção com nível ajustado

Os microdados param em 2024, mas o **indicador nacional de 2025 já foi
divulgado** (66,0%, rede pública — presente na tabela de metas nacionais da
Fase 2). É a única medida disponível do ano projetado, e foi usada para
validar a projeção fora do tempo:

| | Oficial (INEP) | Modelo |
|---|---|---|
| 2024 | 59,2% | 59,3% (out-of-fold) — e 59,2% recalculando dos microdados com `peso_aluno` |
| 2025 | **66,0%** | **61,1%** (projeção) — erro de −4,9 p.p. |

*Leitura*: em 2024 a agregação ponderada e a calibração batem com o número
oficial. Em 2025 o modelo acerta a direção (sobe) mas subestima a
magnitude: ele aprendeu a relação "contexto de 2023 → resultado de 2024" de
um único par de anos e não tem como antecipar uma aceleração nacional
(+6,8 p.p. em 2025, contra +3,3 p.p. em 2024).

*Decisão*: reportar a projeção em dois cenários, sem esconder nenhum:

- **Modelo puro** — cenário pessimista (2.382 de 4.760 municípios abaixo da meta).
- **Nível ajustado** — todas as probabilidades deslocadas por um mesmo valor
  na escala logit até a taxa nacional bater o oficial de 2025
  (`ajustar_nivel()` em `src/evaluation/aplicacao_estrategica.py`). É a
  correção padrão para mudança de prevalência: preserva a ordem entre
  municípios (o ranking de prioridade não muda) e corrige só o nível. Resultado:
  1.757 municípios abaixo da meta.

*Alternativa descartada*: ajustar o modelo usando 2025 e apresentar só o
número ajustado — esconderia que o modelo, sozinho, erra o nível quando a
tendência nacional muda. Esse erro é a principal limitação para uso em
projeção e precisa estar visível.

### D21 — Validação estatística da escolha e modelo desafiante

Três verificações adicionadas depois da escolha, para medir a incerteza e o
custo de ter escolhido o modelo mais simples:

**Regularização da logística otimizada.** Busca em validação cruzada agrupada
sobre `C` ∈ {0,001; 0,01; 0,1; 1; 10}: escolhido `C = 0,01`. Todas as opções
ficam dentro de 0,0001 de ROC-AUC — com 1,48 M alunos e 16 features, o
modelo não está em regime de overfitting (gap treino−validação de 0,004).

**Comparação pareada (mesmos folds).** Random Forest e HistGradientBoosting
superam a logística nos 5 de 5 folds, por +0,003 e +0,004 de ROC-AUC
(intervalos de 95% excluem zero). A diferença é estatisticamente detectável,
mas pequena — 4× menor que a variação entre grupos de municípios — e a
regra de 1 desvio padrão (D14), definida antes, mantém a logística.

**Sensibilidade no teste** (`python -m src.modeling.sensibilidade_modelo`,
`reports/sensibilidade_modelo.json`): o HGB treinado no mesmo treino, no
mesmo teste.

| | Regressão Logística | HistGradientBoosting |
|---|---|---|
| ROC-AUC | 0,628 | 0,634 |
| Brier | 0,229 | 0,228 |
| Lift no top 20% (alunos) | 1,452× | 1,454× |
| Correlação taxa prevista × real (municípios) | 0,73 | 0,77 |
| Erro médio municipal | 9,6 p.p. | 8,9 p.p. |

Os dois priorizam praticamente os mesmos alunos (lift idêntico) e 84% dos
mesmos municípios no top 20% (correlação de 0,93 entre as taxas municipais
previstas pelos dois).

*Decisão*: manter a Regressão Logística como modelo principal. Trocar de
modelo depois de ver o teste invalidaria o teste como medida honesta de
generalização; e, para política pública, um critério explicável por
coeficientes ("cada ponto de taxa de alfabetização no ano anterior aumenta a
chance em X") pesa mais que 0,04 de correlação no agregado. O HGB fica
registrado como **modelo desafiante**: na próxima rodada com dados novos
(2025), os dois são comparados de novo em validação temporal — se o HGB
mantiver a vantagem municipal fora do tempo, passa a ser usado para o
ranking de municípios.

*Alternativa descartada*: escolher pelo maior ROC-AUC médio sem considerar a
variação — o "vencedor" mudaria a cada sorteio de folds.

### D22 — Incerteza do resultado por bootstrap agrupado

O ROC-AUC do teste vem com intervalo de 95% calculado por bootstrap que
reamostra **municípios inteiros** (300 reamostragens): **[0,594; 0,655]**; o
lift do top 20%: **[1,36×; 1,49×]**. Reamostrar alunos um a um daria um
intervalo estreito demais, porque alunos do mesmo município compartilham o
contexto. Os dois intervalos ficam inteiros acima do acaso (0,5 e 1×).
