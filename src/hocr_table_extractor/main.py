# src/hocr_table_extractor/main.py
from __future__ import annotations
from typing import Optional, Tuple, List
import logging
import argparse

from .parser import parse_hocr_words
from .lines import build_lines
from .columns import estimate_columns
from .column_model import infer_numeric_columns_from_lines
from .assign import assign_words_to_columns
from .assign_financial import assign_financial_three_columns
from .assign_dynamic import assign_dynamic
from .rows import merge_lines_into_rows, merge_financial_rows, detect_header_row
from .exporters import rows_to_csv, rows_to_csv_numeric
from .postprocess import fill_missing_labels_and_clean

log = logging.getLogger(__name__)

def hocr_to_csv(hocr_path: str,
                csv_path: str,
                *,
                table_bbox: Optional[Tuple[int, int, int, int]] = None,
                expected_n_cols: Optional[int] = None,
                header_regexes: Optional[List[str]] = None,
                layout: str = "dynamic"
                ) -> None:
    """
    Pipeline: parse → lines → assign (layout) → merge → CSV.
    layout="financial": heurística específica para balances (Cuenta | 2019 | 2018).
    layout="dynamic": infiere columnas numéricas y extrae una columna de texto.
    layout="generic": usa gaps entre palabras para definir columnas.
    """
    log.info("=== HOCR → CSV ===")
    log.info(f"hocr_path={hocr_path}")
    log.info(f"csv_path={csv_path}")
    if table_bbox:
        log.info(f"table_bbox={table_bbox}")
    log.info(f"expected_n_cols={expected_n_cols}, layout={layout}")

    # 1) Parse
    tokens = parse_hocr_words(hocr_path, table_bbox=table_bbox)
    log.info(f"Tokens extraídos: {len(tokens)}")
    if not tokens:
        rows_to_csv([], [], csv_path)
        log.warning("No se extrajeron tokens; se generó CSV vacío.")
        return

    # 2) Líneas
    lines = build_lines(tokens)
    log.info(f"Líneas detectadas: {len(lines)}")

    # 3) Flujos de extracción según el layout
    if layout == "dynamic":
        num_cols, col_names = infer_numeric_columns_from_lines(
            lines, min_sep_px=35, cut_quantile=90.0, pad_px=24
        )
        log.info(f"Columnas numéricas detectadas ({len(num_cols)}): {num_cols}")
        if col_names:
            log.info(f"Nombres detectados: {col_names}")

        records = assign_dynamic(lines, num_cols)
        log.info(f"Registros (dynamic): {len(records)}")

        rows = merge_financial_rows(records)
        log.info(f"Filas fusionadas (pre post-proceso): {len(rows)}")
        if not rows:
            rows_to_csv([], [], csv_path); log.warning("CSV vacío."); return

        rows = fill_missing_labels_and_clean(rows, label_for_subtotals=True, normalize_dash_zero=True)
        log.info(f"Filas tras post-proceso: {len(rows)}")

        n = max((len(r) for r in rows), default=1)
        if col_names and len(col_names) == (n - 1):
            header = ["Cuenta"] + col_names
        else:
            header = ["Cuenta"] + [f"Valor_{i+1}" for i in range(n-1)]

        body = [r + [""]*(n-len(r)) if len(r)<n else r[:n] for r in rows]

        rows_to_csv(body, header, csv_path)
        num_path = csv_path.replace(".csv", ".num.csv")
        rows_to_csv_numeric(body, header, num_path)

        log.info(f"✔ CSV escrito en: {csv_path}")
        log.info(f"✔ CSV numérico escrito en: {num_path}")

    elif layout == "financial":
        records = assign_financial_three_columns(lines)
        log.info(f"Registros (financial): {len(records)}")
        rows = merge_financial_rows(records)
        log.info(f"Filas fusionadas: {len(rows)}")
        header = ["Cuenta", "Valor_1", "Valor_2"]
        rows_to_csv(rows, header, csv_path)
        log.info(f"✔ CSV escrito en: {csv_path}")

    elif layout == "generic":
        columns = estimate_columns(lines, expected_n_cols=expected_n_cols)
        log.info(f"Columnas estimadas ({len(columns)}): {columns}")
        records = assign_words_to_columns(lines, columns)
        rows = merge_lines_into_rows(records, lines)
        header, body = detect_header_row(rows, header_regexes=header_regexes)
        if header is None:
            ncols = max((len(r) for r in rows), default=0)
            header = [f"col_{i+1}" for i in range(ncols)]
            body = rows
        rows_to_csv(body, header, csv_path)
        log.info(f"✔ CSV escrito en: {csv_path}")

    else:
        log.error(f"Layout desconocido: '{layout}'. Use 'financial', 'dynamic' o 'generic'.")
