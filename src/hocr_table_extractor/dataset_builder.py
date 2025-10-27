from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .columns import estimate_columns
from .lines import build_lines
from .parser import parse_hocr_words
from .rows import detect_header_row
from .structures import Token

log = logging.getLogger(__name__)


@dataclass
class LayoutLMExample:
    image_path: str
    hocr_path: str
    words: List[str]
    bboxes: List[List[int]]
    labels: List[str]
    row_ids: List[int]
    col_ids: List[int]
    is_header: List[bool]
    table_header: List[str]
    table_rows: List[List[str]]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_box(box: List[int], width: int, height: int) -> List[int]:
    w = max(width, 1)
    h = max(height, 1)
    x1, y1, x2, y2 = box
    return [
        int(max(0, min(1000, round(1000 * x1 / w)))),
        int(max(0, min(1000, round(1000 * y1 / h)))),
        int(max(0, min(1000, round(1000 * x2 / w)))),
        int(max(0, min(1000, round(1000 * y2 / h)))),
    ]


def _compute_row_intervals(lines: List[Dict[str, object]]) -> List[Tuple[int, int]]:
    tokens = [tok for ln in lines for tok in ln["tokens"] if getattr(tok, "text", None)]
    if not tokens:
        return []

    y_min = min(t.y1 for t in tokens)
    y_max = max(t.y2 for t in tokens)
    height = max(1, y_max - y_min)

    profile = np.zeros(height, dtype=int)
    for tok in tokens:
        start = max(0, tok.y1 - y_min)
        end = max(start + 1, tok.y2 - y_min)
        profile[start:end] += 1

    zero_indices = np.where(profile == 0)[0]
    if len(zero_indices) == 0:
        return [(y_min, y_max)]

    gaps = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    cuts = [y_min]
    for gap in gaps:
        if len(gap) > 2:
            cuts.append(y_min + int(gap.mean()))
    cuts.append(y_max)
    cuts = sorted(list(dict.fromkeys(cuts)))

    intervals: List[Tuple[int, int]] = []
    for top, bottom in zip(cuts, cuts[1:]):
        if bottom - top > 5:
            intervals.append((top, bottom))
    return intervals or [(y_min, y_max)]


def _assign_lines_to_rows(
    lines: List[Dict[str, object]],
    row_intervals: List[Tuple[int, int]],
) -> List[int]:
    if not row_intervals:
        return [0 for _ in lines]

    centers = [(top + bottom) / 2.0 for (top, bottom) in row_intervals]
    mapping: List[int] = []
    for ln in lines:
        y_top = ln["bbox"][1]
        y_bot = ln["bbox"][3]
        yc = (y_top + y_bot) / 2.0
        idx = None
        for i, (top, bottom) in enumerate(row_intervals):
            if top <= yc < bottom:
                idx = i
                break
        if idx is None:
            distances = [abs(c - yc) for c in centers]
            idx = int(np.argmin(distances)) if distances else 0
        mapping.append(idx)
    return mapping


def _find_column_index(token: Token, intervals: List[Tuple[int, int]]) -> Optional[int]:
    if not intervals:
        return None
    xc = token.xc
    for i, (x1, x2) in enumerate(intervals):
        if x1 <= xc <= x2:
            return i
    centers = [((x1 + x2) / 2.0) for (x1, x2) in intervals]
    distances = [abs(c - xc) for c in centers]
    if not distances:
        return None
    return int(np.argmin(distances))


def _aggregate_rows(
    lines: List[Dict[str, object]],
    line_to_row: List[int],
    column_intervals: List[Tuple[int, int]],
) -> Tuple[List[List[str]], Dict[int, List[List[str]]], List[Tuple[Token, int, Optional[int]]]]:
    col_count = len(column_intervals)
    row_count = max(line_to_row) + 1 if line_to_row else 0
    rows_tokens: Dict[int, List[List[str]]] = {
        idx: [[] for _ in range(col_count)] for idx in range(row_count)
    }
    token_records: List[Tuple[Token, int, Optional[int]]] = []

    for ln, row_idx in zip(lines, line_to_row):
        for tok in ln["tokens"]:
            col_idx = _find_column_index(tok, column_intervals)
            token_records.append((tok, row_idx, col_idx))
            if row_idx in rows_tokens and col_idx is not None and col_idx < col_count:
                rows_tokens[row_idx][col_idx].append(tok.text)

    grid_rows: List[List[str]] = []
    for row_idx in range(row_count):
        cols = [" ".join(filter(None, rows_tokens[row_idx][j])).strip() for j in range(col_count)]
        grid_rows.append(cols)
    return grid_rows, rows_tokens, token_records


def _detect_header_index(
    grid_rows: List[List[str]],
    header_regexes: Optional[Sequence[str]],
) -> Tuple[Optional[int], List[str], List[List[str]]]:
    if not grid_rows:
        return None, [], []
    header_row, body_rows = detect_header_row(
        grid_rows,
        header_regexes=list(header_regexes) if header_regexes else None,
    )
    header_idx = None
    if header_row:
        for idx, row in enumerate(grid_rows):
            if row == header_row:
                header_idx = idx
                break
    return header_idx, header_row or [], body_rows


def build_layoutlm_example(
    *,
    image_path: str,
    hocr_path: str,
    table_bbox: Optional[Tuple[int, int, int, int]] = None,
    expected_n_cols: Optional[int] = None,
    header_regexes: Optional[Sequence[str]] = None,
    max_columns: int = 6,
) -> LayoutLMExample:
    """
    Construye un ejemplo etiquetado usando el layout 'generic' como maestro.
    La salida contiene listas sincronizadas de palabras, bounding boxes normalizados
    y etiquetas pensadas para fine-tuning de LayoutLM.
    """
    log.info("Generando ejemplo LayoutLM: image=%s hocr=%s", image_path, hocr_path)

    tokens = parse_hocr_words(hocr_path, table_bbox=table_bbox)
    if not tokens:
        raise ValueError(f"No se encontraron tokens en {hocr_path}")

    lines = build_lines(tokens)
    if not lines:
        raise ValueError("No se pudieron construir líneas a partir de los tokens.")

    column_intervals = estimate_columns(
        lines,
        expected_n_cols=expected_n_cols,
    )
    if not column_intervals:
        raise ValueError("No se detectaron columnas. Ajuste expected_n_cols o bbox.")

    row_intervals = _compute_row_intervals(lines)
    line_to_row = _assign_lines_to_rows(lines, row_intervals)

    grid_rows, rows_tokens, token_records = _aggregate_rows(
        lines,
        line_to_row,
        column_intervals,
    )

    header_idx, header_row, body_rows = _detect_header_index(
        grid_rows,
        header_regexes,
    )

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No se encontró la imagen {image_path}") from exc

    words: List[str] = []
    boxes: List[List[int]] = []
    labels: List[str] = []
    row_ids: List[int] = []
    col_ids: List[int] = []
    is_header_flags: List[bool] = []

    for tok, row_idx, col_idx in token_records:
        words.append(tok.text)
        boxes.append([tok.x1, tok.y1, tok.x2, tok.y2])
        row_ids.append(row_idx if row_idx is not None else -1)
        col_ids.append(col_idx if col_idx is not None else -1)
        is_header = header_idx is not None and row_idx == header_idx
        is_header_flags.append(is_header)

        if col_idx is None or col_idx < 0 or col_idx >= max_columns:
            label = "OTHER"
        else:
            prefix = "HEADER" if is_header else "BODY"
            label = f"{prefix}_COL_{col_idx}"
        labels.append(label)

    norm_boxes = [_normalize_box(box, width, height) for box in boxes]

    metadata: Dict[str, object] = {
        "column_intervals": column_intervals,
        "row_intervals": row_intervals,
        "header_index": header_idx,
        "rows_tokens": {
            str(idx): [[txt for txt in cell] for cell in rows_tokens[idx]]
            for idx in rows_tokens
        },
    }

    example = LayoutLMExample(
        image_path=image_path,
        hocr_path=hocr_path,
        words=words,
        bboxes=norm_boxes,
        labels=labels,
        row_ids=row_ids,
        col_ids=col_ids,
        is_header=is_header_flags,
        table_header=header_row,
        table_rows=body_rows,
        metadata=metadata,
    )
    return example
