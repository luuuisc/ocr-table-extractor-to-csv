# src/hocr_table_extractor/main.py
from __future__ import annotations
from typing import Optional, Tuple, List
import logging

from .parser import parse_hocr_words
from .lines import build_lines
from .columns import estimate_columns
from .assign_financial import assign_financial_three_columns
from .rows import merge_lines_into_rows, merge_financial_rows, detect_header_row
from .exporters import rows_to_csv

log = logging.getLogger(__name__)

def hocr_to_csv(hocr_path: str,
                csv_path: str,
                *,
                table_bbox: Optional[Tuple[int, int, int, int]] = None,
                expected_n_cols: Optional[int] = None,
                header_regexes: Optional[List[str]] = None,
                col_gap_quantile: float = 92.0,
                row_merge_factor: float = 1.25,
                layout: str = "financial"
                ) -> None:
    """
    Pipeline: parse → lines → assign (layout) → merge → CSV.
    layout="financial": heurística específica para balances (Cuenta | 2019 | 2018).
    """
    log.info("=== HOCR → CSV ===")
    log.info(f"hocr_path={hocr_path}")
    log.info(f"csv_path={csv_path}")
    if table_bbox:
        log.info(f"table_bbox={table_bbox}")
    log.info(f"expected_n_cols={expected_n_cols}, col_gap_quantile={col_gap_quantile}, row_merge_factor={row_merge_factor}, layout={layout}")

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

    if layout == "financial":
        # 3) Asignación específica balances
        records = assign_financial_three_columns(lines, label_col_name="Cuenta", newest_on_right=True)
        log.info(f"Registros (modo financial): {len(records)}")

        # 4) Merge financiero (inteligente)
        rows = merge_financial_rows(records, row_merge_factor=max(row_merge_factor, 1.30))
        log.info(f"Filas fusionadas: {len(rows)}")
        if not rows:
            rows_to_csv([], [], csv_path)
            log.warning("No se construyeron filas; se generó CSV vacío.")
            return

        # 5) Encabezado fijo de 3 columnas
        header = ["Cuenta", "2019", "2018"]
        body = rows
        # filtra filas totalmente vacías
        body = [r for r in body if any((r + ["", "", ""])[:3])]

        rows_to_csv(body, header, csv_path)
        log.info(f"✔ CSV escrito en: {csv_path}")
        return

    # --- Alternativa genérica (si se pide layout="generic") ---
    columns = estimate_columns(lines, col_gap_quantile=col_gap_quantile, expected_n_cols=expected_n_cols)
    log.info(f"Columnas estimadas ({len(columns)}): {columns}")
    from .assign import assign_words_to_columns
    records = assign_words_to_columns(lines, columns)
    rows = merge_lines_into_rows(records, row_merge_factor=row_merge_factor)
    header, body = detect_header_row(rows, header_regexes=header_regexes)
    if header is None:
        ncols = max((len(r) for r in rows), default=0)
        header = [f"col_{i+1}" for i in range(ncols)]
        body = rows
    rows_to_csv(body, header, csv_path)
    log.info(f"✔ CSV escrito en: {csv_path}")
