# data/

| Pasta | Conteúdo | Versionado? | Gerado por |
|---|---|---|---|
| `raw/` | CSVs do INEP (Base dos Dados), diretórios territoriais, IDHM (Atlas) e `raw_metadados.xlsx` | Sim, exceto `br_inep_avaliacao_alfabetizacao_alunos.csv` (~214 MB, acima do limite do GitHub) | Download manual — ver "Como Reproduzir" no [README](../README.md#como-reproduzir) |
| `bronze/` | Dados brutos em Parquet, uma tabela por fonte (inclui as versões anuais das metas) | Não | `python pipeline/executar_pipeline.py` |
| `silver/` | Dados limpos e integrados (inclui `alunos`, base do alvo do modelo) | Não | idem |
| `gold/` | Tabelas analíticas (indicador municipal, evolução temporal, metas x resultado) | Não | idem, só se passar no gate de qualidade |
| `gold_rejeitada/` | Gold que reprovou no gate de qualidade, para inspeção | Não | idem |
| `monitoramento/` | Alertas de execução e de qualidade (`alertas.jsonl`) | Não | idem |
| `modelagem/` | Predições out-of-fold de 2024 e projeção de 2025, por aluno | Não | `python -m src.modeling.treinar` |

O dataset de modelagem (1 linha por aluno, 16 features) não é gravado em
disco: é montado em memória por
[`src/preprocessing/dataset.py`](../src/preprocessing/dataset.py) a partir de
`bronze/`, `silver/` e `gold/`, para que qualquer mudança nas camadas se
reflita no modelo sem passo intermediário a sincronizar.
