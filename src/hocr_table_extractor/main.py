from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .assign import assign_words_to_columns
from .assign_dynamic import assign_dynamic
from .assign_financial import assign_financial_three_columns
from .column_model import infer_numeric_columns_from_lines
from .columns import estimate_columns
from .exporters import rows_to_csv, rows_to_csv_numeric
from .layout_professional import extract_professional_layout
from .layout_transformers import (
    DEFAULT_MODEL_ID as TRANSFORMER_DEFAULT_MODEL,
    DEFAULT_OCR_LANG as TRANSFORMER_DEFAULT_LANG,
    MAX_MODEL_COLUMNS as TRANSFORMER_MAX_COLUMNS,
    extract_transformers_layout,
)
from .lines import build_lines
from .parser import parse_hocr_words
from .postprocess import fill_missing_labels_and_clean
from .rows import detect_header_row, merge_financial_rows, merge_lines_into_rows

log = logging.getLogger(__name__)


def _ensure_parent_dir(csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)


def _numeric_variant_path(csv_path: str) -> Path:
    path = Path(csv_path)
    suffix = ".csv"
    if path.suffix.lower() != suffix:
        return path.with_name(f"{path.name}.num.csv")
    return path.with_name(f"{path.stem}.num.csv")


def _write_empty_csv(csv_path: str) -> None:
    _ensure_parent_dir(csv_path)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("")


def _resolve_header(num_columns: int, candidate_names: Optional[Sequence[str]]) -> Sequence[str]:
    label = "Cuenta"
    numeric_names = []
    for idx in range(num_columns):
        if candidate_names and idx < len(candidate_names) and candidate_names[idx]:
            numeric_names.append(str(candidate_names[idx]))
        else:
            numeric_names.append(f"Valor_{idx + 1}")
    return [label, *numeric_names]


def hocr_to_csv(
    hocr_path: Optional[str],
    csv_path: str,
    *,
    image_path: Optional[str] = None,
    layout: str = "dynamic",
    table_bbox: Optional[Tuple[int, int, int, int]] = None,
    expected_n_cols: Optional[int] = None,
    header_regexes: Optional[Sequence[str]] = None,
    transformer_model: Optional[str] = None,
    transformer_ocr_lang: Optional[str] = None,
    transformer_max_columns: Optional[int] = None,
) -> None:
    """
    Orquesta la reconstrucción de tablas según el layout seleccionado.
    Mantiene intactos los layouts basados en HOCR y añade el modo `transformers`.
    """
    layout = (layout or "dynamic").lower()
    log.info("Layout seleccionado: %s", layout)

    if layout == "transformers":
        if not image_path:
            raise ValueError("image_path es requerido para el layout 'transformers'.")
        _ensure_parent_dir(csv_path)
        extract_transformers_layout(
            image_path=image_path,
            csv_path=csv_path,
            table_bbox=table_bbox,
            model_id=transformer_model or TRANSFORMER_DEFAULT_MODEL,
            ocr_lang=transformer_ocr_lang or TRANSFORMER_DEFAULT_LANG,
            max_columns=transformer_max_columns
            or expected_n_cols
            or TRANSFORMER_MAX_COLUMNS,
            expected_n_cols=expected_n_cols,
            header_regexes=list(header_regexes) if header_regexes else None,
        )
        return

    if not hocr_path:
        raise ValueError("hocr_path es requerido para layouts basados en HOCR.")

    log.info("Parseando HOCR desde: %s", hocr_path)
    tokens = parse_hocr_words(hocr_path, table_bbox=table_bbox)
    if not tokens:
        log.warning("No se encontraron tokens en el HOCR. Se generará un CSV vacío.")
        _write_empty_csv(csv_path)
        return

    lines = build_lines(tokens)
    if not lines:
        log.warning("No se pudieron construir líneas. Se generará un CSV vacío.")
        _write_empty_csv(csv_path)
        return

    if layout == "financial":
        log.info("Ejecutando layout 'financial'.")
        records = assign_financial_three_columns(lines)
        rows = merge_financial_rows(records)
        rows = fill_missing_labels_and_clean(rows)
        header = ["Cuenta", "Valor_1", "Valor_2"]
        _ensure_parent_dir(csv_path)
        rows_to_csv(rows, header, csv_path)
        return

    if layout == "dynamic":
        log.info("Ejecutando layout 'dynamic'.")
        intervals, header_names = infer_numeric_columns_from_lines(lines)
        log.debug("Intervalos numéricos detectados: %s", intervals)
        records = assign_dynamic(lines, intervals)
        rows = merge_financial_rows(records)
        if not rows:
            log.warning("No se generaron filas para el layout 'dynamic'. CSV vacío.")
            _write_empty_csv(csv_path)
            return
        num_cols = max(len(row) for row in rows) - 1 if rows else 0
        header = list(_resolve_header(max(num_cols, 0), header_names))
        _ensure_parent_dir(csv_path)
        rows_to_csv(rows, header, csv_path)
        numeric_csv = _numeric_variant_path(csv_path)
        rows_to_csv_numeric(rows, header, str(numeric_csv))
        log.info("CSV numérico adicional escrito en: %s", numeric_csv)
        return

    if layout == "generic":
        log.info("Ejecutando layout 'generic'.")
        intervals = estimate_columns(lines, expected_n_cols=expected_n_cols)
        records = assign_words_to_columns(lines, intervals)
        grid_rows = merge_lines_into_rows(records, lines)
        header_row, body_rows = detect_header_row(
            grid_rows,
            header_regexes=list(header_regexes) if header_regexes else None,
        )
        header = header_row or []
        _ensure_parent_dir(csv_path)
        rows_to_csv(body_rows, header, csv_path)
        return

    if layout == "professional":
        log.info("Ejecutando layout 'professional'.")
        extract_professional_layout(
            hocr_path=hocr_path,
            csv_path=csv_path,
            table_bbox=table_bbox,
        )
        return

    raise ValueError(f"Layout desconocido: {layout!r}")
