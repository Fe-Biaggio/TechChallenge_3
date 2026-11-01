"""
Inferência com o modelo salvo — o uso "em produção": a cada nova divulgação
do INEP, pontuar o ano seguinte sem re-treinar.

Carrega models/modelo_final.joblib (pré-processamento + modelo num único
Pipeline), monta as features do ano-alvo a partir das camadas da pipeline
(contexto municipal do ano anterior, meta do ano) e grava:

  - taxa prevista por município (rede municipal) vs. meta do ano-alvo
  - risco de não atingir a meta

Opcionalmente ajusta o nível geral a uma taxa nacional conhecida (D20 em
docs/decisoes_analiticas.md) — útil quando o resultado nacional do ano já
saiu, mas os microdados municipais ainda não.

Uso:
    python -m src.modeling.prever                                  # 2025, composição de 2024
    python -m src.modeling.prever --ano-alvo 2025 --taxa-nacional 66.0
    python -m src.modeling.prever --saida reports/minha_projecao.csv
"""
import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation import aplicacao_estrategica as estrategia  # noqa: E402
from src.preprocessing.dataset import ANO_TREINO, TODAS_FEATURES, montar_dataset_projecao  # noqa: E402

MODELO = ROOT / "models" / "modelo_final.joblib"


def prever(ano_alvo: int, ano_roster: int, taxa_nacional: float = None):
    """Probabilidade por aluno e agregação por município para `ano_alvo`."""
    modelo = joblib.load(MODELO)
    df = montar_dataset_projecao(ano_alvo, ano_roster)
    df["proba_projecao"] = modelo.predict_proba(df[TODAS_FEATURES])[:, 1]
    delta = None
    if taxa_nacional is not None:
        df["proba_projecao"], delta = estrategia.ajustar_nivel(df, "proba_projecao", taxa_nacional)
    municipios, por_regiao = estrategia.projecao_meta(df)
    return df, municipios, por_regiao, delta


def main(argv=None):
    """Linha de comando: grava a previsão por município e imprime o resumo por região."""
    parser = argparse.ArgumentParser(description="Projeção de alfabetização por município com o modelo salvo")
    parser.add_argument("--ano-alvo", type=int, default=ANO_TREINO + 1)
    parser.add_argument("--ano-roster", type=int, default=ANO_TREINO,
                        help="ano cuja lista de alunos/escolas aproxima a coorte do ano-alvo (D18)")
    parser.add_argument("--taxa-nacional", type=float, default=None,
                        help="taxa nacional oficial do ano-alvo, em %%, para ajustar o nível (D20)")
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args(argv)

    df, municipios, por_regiao, delta = prever(args.ano_alvo, args.ano_roster, args.taxa_nacional)
    saida = args.saida or ROOT / "reports" / f"previsao_{args.ano_alvo}.csv"
    municipios.to_csv(saida, index=False)

    print(f"[prever] {len(df):,} alunos pontuados | taxa nacional prevista (rede pública): "
          f"{estrategia.taxa_nacional(df, 'proba_projecao')}%"
          + (f" | nível ajustado (deslocamento logit {delta})" if delta is not None else ""))
    print(f"[prever] {int(municipios.risco_nao_atingir.sum())} de {len(municipios)} municípios abaixo da meta "
          f"de {args.ano_alvo} — gravado em {saida}")
    print(por_regiao.to_string(index=False))


if __name__ == "__main__":
    main()
