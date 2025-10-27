from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .evaluation import evaluate_tables, write_report

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa un CSV predicho contra una referencia y calcula métricas (text accuracy, MSE, RMSE, R2)."
    )
    parser.add_argument("--reference", required=True, help="CSV de referencia (ground truth).")
    parser.add_argument("--predicted", required=True, help="CSV generado por el layout a evaluar.")
    parser.add_argument("--numeric-columns", nargs="+", help="Lista opcional de columnas numéricas a evaluar. Si se omite, se infieren automáticamente.")
    parser.add_argument("--report", help="Ruta opcional para guardar un reporte CSV con las métricas.")
    parser.add_argument("--json", help="Ruta opcional para guardar métricas en JSON.")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(levelname)s - %(message)s")

    evaluation = evaluate_tables(
        reference_csv=args.reference,
        predicted_csv=args.predicted,
        numeric_columns=args.numeric_columns,
    )

    log.info("Text accuracy: %.4f (%d/%d)", evaluation.text_accuracy, evaluation.matched_cells, evaluation.total_cells)
    for metric in evaluation.numeric_by_column:
        log.info("Numeric column %s -> MSE: %.6f RMSE: %.6f R2: %.6f (n=%d)", metric.column, metric.mse, metric.rmse, metric.r2, metric.n)
    if evaluation.numeric_overall:
        overall = evaluation.numeric_overall
        log.info("Numeric overall -> MSE: %.6f RMSE: %.6f R2: %.6f (n=%d)", overall.mse, overall.rmse, overall.r2, overall.n)

    if args.report:
        write_report(evaluation, args.report)
        log.info("Reporte CSV guardado en %s", args.report)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(evaluation.to_dict(), fh, indent=2)
        log.info("Reporte JSON guardado en %s", args.json)


if __name__ == "__main__":
    main()
