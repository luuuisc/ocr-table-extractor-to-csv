from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cleaners import process_grid_data
from .columns import estimate_columns
from .assign import assign_words_to_columns
from .exporters import rows_to_csv
from .grid_builder import build_grid_from_words
from .lines import build_lines
from .rows import detect_header_row, merge_lines_into_rows
from .spatial import BBox, SpatialWord
from .structures import Token

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "microsoft/layoutlmv3-base"
DEFAULT_OCR_LANG = "eng"
OCR_CONFIDENCE_THRESHOLD = 60
MAX_MODEL_COLUMNS = 6
HEADER_PREFIX = "HEADER_COL_"
BODY_PREFIX = "BODY_COL_"
OTHER_LABEL = "OTHER"


@dataclass
class TokenPrediction:
    text: str
    bbox: List[int]
    label: str
    column: Optional[int]
    kind: str
    x_center: float
    y_center: float
    height: int
    token_ref: Optional[Token] = None


@dataclass
class RowGroup:
    index: int
    tokens_by_col: Dict[int, List[TokenPrediction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _y_values: List[float] = field(default_factory=list)
    is_header_row: bool = False

    def add(self, token: TokenPrediction) -> None:
        col = 0 if token.column is None or token.column < 0 else token.column
        self.tokens_by_col[col].append(token)
        self._y_values.append(token.y_center)
        if token.kind == "header":
            self.is_header_row = True

    @property
    def y_center(self) -> float:
        if not self._y_values:
            return 0.0
        return sum(self._y_values) / len(self._y_values)


def _run_tesseract_ocr(
    image,
    *,
    table_bbox: Optional[Tuple[int, int, int, int]] = None,
    lang: str = DEFAULT_OCR_LANG,
) -> Tuple[List[str], List[List[int]]]:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - import guarded por entorno
        raise RuntimeError(
            "pytesseract no está instalado. Instale requirements.txt y Tesseract."
        ) from exc

    ocr_data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        lang=lang,
    )
    n_boxes = len(ocr_data["level"])
    words: List[str] = []
    boxes: List[List[int]] = []

    tx0 = ty0 = tx1 = ty1 = None
    if table_bbox:
        tx0, ty0, tx1, ty1 = table_bbox

    for i in range(n_boxes):
        try:
            conf = int(float(ocr_data["conf"][i]))
        except Exception:
            conf = -1
        if conf <= OCR_CONFIDENCE_THRESHOLD:
            continue

        text = (ocr_data["text"][i] or "").strip()
        if not text:
            continue

        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        bbox = [x, y, x + w, y + h]

        if table_bbox and not (
            bbox[0] >= tx0  # type: ignore[operator]
            and bbox[1] >= ty0
            and bbox[2] <= tx1
            and bbox[3] <= ty1
        ):
            continue

        words.append(text)
        boxes.append(bbox)

    return words, boxes


@lru_cache(maxsize=1)
def _load_layoutlmv3(model_id: str = DEFAULT_MODEL_ID):
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "transformers y torch son requeridos para el layout 'transformers'."
        ) from exc

    processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_id)
    return processor, model


def _predict_labels(processor, model, image, words: List[str], boxes: List[List[int]]):
    import torch

    encoding = processor(image, words, boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids()

    collapsed_labels: List[str] = []
    prev_word_idx = -1
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        label = model.config.id2label[predictions[i]]
        collapsed_labels.append(label)
        prev_word_idx = word_idx
    return collapsed_labels


def _parse_prediction_label(label: str) -> Tuple[str, Optional[int]]:
    if label.startswith(HEADER_PREFIX):
        try:
            column = int(label.replace(HEADER_PREFIX, ""))
        except ValueError:
            column = None
        return "header", column
    if label.startswith(BODY_PREFIX):
        try:
            column = int(label.replace(BODY_PREFIX, ""))
        except ValueError:
            column = None
        return "body", column
    return "other", None


def _build_predictions(words: List[str], boxes: List[List[int]], labels: List[str]) -> List[TokenPrediction]:
    predictions: List[TokenPrediction] = []
    for word, box, label in zip(words, boxes, labels):
        kind, column = _parse_prediction_label(label)
        x1, y1, x2, y2 = box
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        height = max(1, y2 - y1)
        predictions.append(
            TokenPrediction(
                text=word,
                bbox=box,
                label=label,
                column=column,
                kind=kind,
                x_center=xc,
                y_center=yc,
                height=height,
            )
        )
    return predictions


def _derive_column_intervals(
    tokens: List[TokenPrediction],
    *,
    max_columns: Optional[int],
) -> List[Tuple[int, int]]:
    labeled_columns: Dict[int, List[TokenPrediction]] = defaultdict(list)
    for token in tokens:
        if token.column is not None and token.column >= 0:
            labeled_columns[token.column].append(token)

    intervals: List[Tuple[int, int]] = []

    if labeled_columns:
        for col in sorted(labeled_columns.keys()):
            xs_left = [t.bbox[0] for t in labeled_columns[col]]
            xs_right = [t.bbox[2] for t in labeled_columns[col]]
            if xs_left and xs_right:
                pad = 3
                intervals.append(
                    (
                        int(min(xs_left) - pad),
                        int(max(xs_right) + pad),
                    )
                )

    def _vertical_profile_intervals() -> List[Tuple[int, int]]:
        if not tokens:
            return []
        x_min = min(t.bbox[0] for t in tokens)
        x_max = max(t.bbox[2] for t in tokens)
        width = max(1, x_max - x_min)
        profile = np.zeros(width, dtype=int)
        for token in tokens:
            start = max(0, token.bbox[0] - x_min)
            end = max(start + 1, token.bbox[2] - x_min)
            profile[start:end] += 1
        zero_indices = np.where(profile == 0)[0]
        if len(zero_indices) == 0:
            return [(int(x_min), int(x_max))]
        gaps = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
        cuts = [x_min]
        for gap in gaps:
            if len(gap) > 3:
                cuts.append(x_min + int(gap.mean()))
        cuts.append(x_max)
        cuts = sorted(list(dict.fromkeys(cuts)))
        intervals_local: List[Tuple[int, int]] = []
        for left, right in zip(cuts, cuts[1:]):
            if right - left > 5:
                intervals_local.append((int(left), int(right)))
        if not intervals_local:
            intervals_local.append((int(x_min), int(x_max)))
        return intervals_local

    if not intervals:
        intervals = _vertical_profile_intervals()

    if max_columns:
        intervals = intervals[:max_columns]

    # Si seguimos con menos de 2 columnas y el usuario espera más, forzamos un perfil vertical.
    min_required = 2 if max_columns and max_columns > 1 else 1
    if len(intervals) < min_required:
        intervals = _vertical_profile_intervals()
        if max_columns:
            intervals = intervals[:max_columns]

    if max_columns and intervals:
        while len(intervals) < max_columns:
            widths = [right - left for left, right in intervals]
            widest_idx = int(np.argmax(widths))
            left, right = intervals[widest_idx]
            if right - left <= 6:
                break  # no margen para dividir
            mid = (left + right) // 2
            intervals = (
                intervals[:widest_idx]
                + [(left, mid), (mid, right)]
                + intervals[widest_idx + 1 :]
            )
        if len(intervals) > max_columns:
            intervals = intervals[:max_columns]

    intervals = sorted(intervals, key=lambda iv: iv[0])

    return intervals


def _assign_columns_from_intervals(
    tokens: List[TokenPrediction],
    intervals: List[Tuple[int, int]],
) -> None:
    if not intervals:
        return
    centers = [((x1 + x2) / 2.0) for (x1, x2) in intervals]
    for token in tokens:
        if token.column is not None and token.column >= 0 and token.column < len(intervals):
            continue
        xc = token.x_center
        idx = None
        for i, (x1, x2) in enumerate(intervals):
            if x1 <= xc <= x2:
                idx = i
                break
        if idx is None and centers:
            distances = [abs(c - xc) for c in centers]
            idx = int(np.argmin(distances))
        token.column = idx if idx is not None else 0


def _compute_row_intervals_from_predictions(tokens: List[TokenPrediction]) -> List[Tuple[int, int]]:
    if not tokens:
        return []
    y_min = min(t.bbox[1] for t in tokens)
    y_max = max(t.bbox[3] for t in tokens)
    height = max(1, y_max - y_min)
    profile = np.zeros(height, dtype=int)
    for token in tokens:
        start = max(0, token.bbox[1] - y_min)
        end = max(start + 1, token.bbox[3] - y_min)
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


def _assign_tokens_to_rows(
    tokens: List[TokenPrediction],
    intervals: List[Tuple[int, int]],
) -> Dict[int, int]:
    if not intervals:
        return {i: 0 for i in range(len(tokens))}
    centers = [(top + bottom) / 2.0 for (top, bottom) in intervals]
    assignments: Dict[int, int] = {}
    for idx, token in enumerate(tokens):
        yc = token.y_center
        row_idx = None
        for i, (top, bottom) in enumerate(intervals):
            if top <= yc < bottom:
                row_idx = i
                break
        if row_idx is None:
            distances = [abs(c - yc) for c in centers]
            row_idx = int(np.argmin(distances)) if distances else 0
        assignments[idx] = row_idx
    return assignments


def _compose_table_from_predictions(
    predictions: List[TokenPrediction],
    *,
    max_columns: int = MAX_MODEL_COLUMNS,
) -> Tuple[List[str], List[List[str]]]:
    if not predictions:
        return [], []

    usable_tokens = [p for p in predictions if p.kind in {"header", "body"}]
    if not usable_tokens:
        usable_tokens = [p for p in predictions if p.kind != "other"]
    if not usable_tokens:
        return [], []

    intervals = _derive_column_intervals(
        usable_tokens,
        max_columns=max_columns,
    )
    if not intervals:
        x_min = min(t.bbox[0] for t in usable_tokens)
        x_max = max(t.bbox[2] for t in usable_tokens)
        intervals = [(int(x_min), int(x_max))]
    _assign_columns_from_intervals(usable_tokens, intervals)

    row_intervals = _compute_row_intervals_from_predictions(usable_tokens)
    assignments = _assign_tokens_to_rows(usable_tokens, row_intervals)

    rows_map: Dict[int, RowGroup] = {}
    for idx, token in enumerate(usable_tokens):
        row_idx = assignments.get(idx, 0)
        group = rows_map.setdefault(row_idx, RowGroup(index=row_idx))
        group.add(token)

    if not rows_map:
        return [], []

    # Ordenar filas por coordenada vertical
    sorted_groups = sorted(rows_map.values(), key=lambda g: g.y_center)

    detected_columns = set()
    for group in sorted_groups:
        detected_columns.update(group.tokens_by_col.keys())

    if not detected_columns:
        return [], []

    max_col = max(detected_columns)
    if max_columns:
        max_col = min(max_col, max_columns - 1)

    num_cols = max_col + 1
    for token in usable_tokens:
        if token.column is None:
            token.column = 0
        elif token.column > max_col:
            token.column = max_col

    header_row = next((g for g in sorted_groups if g.is_header_row), None)
    header: List[str] = []
    for col in range(num_cols):
        tokens = header_row.tokens_by_col.get(col, []) if header_row else []
        tokens = sorted(tokens, key=lambda t: t.x_center)
        text = " ".join(t.text for t in tokens).strip()
        if not text:
            text = "Cuenta" if col == 0 else f"Valor_{col}"
        header.append(text)

    rows: List[List[str]] = []
    for group in sorted_groups:
        if group is header_row:
            continue
        row_values: List[str] = []
        for col in range(num_cols):
            tokens = group.tokens_by_col.get(col, [])
            tokens = sorted(tokens, key=lambda t: t.x_center)
            text = " ".join(t.text for t in tokens).strip()
            row_values.append(text)
        rows.append(row_values)

    rows = process_grid_data(rows)
    return header, rows


def extract_transformers_layout(
    image_path: str,
    csv_path: str,
    *,
    table_bbox: Optional[Tuple[int, int, int, int]] = None,
    model_id: str = DEFAULT_MODEL_ID,
    ocr_lang: str = DEFAULT_OCR_LANG,
    max_columns: Optional[int] = None,
    expected_n_cols: Optional[int] = None,
    header_regexes: Optional[Sequence[str]] = None,
) -> None:
    """Reconstruye una tabla usando LayoutLMv3 y OCR desde la imagen de entrada."""
    log.info("=== Layout basado en Transformers ===")
    log.info("Modelo: %s", model_id)
    log.info("Imagen: %s", image_path)
    log.info("CSV destino: %s", csv_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow es requerido para el layout 'transformers'. Instale requirements.txt."
        ) from exc

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    log.info("Imagen cargada con tamaño: %s", image.size)

    words, boxes = _run_tesseract_ocr(
        image,
        table_bbox=table_bbox,
        lang=ocr_lang,
    )
    log.info("Palabras válidas extraídas por OCR: %d", len(words))

    if not words:
        log.warning("OCR no produjo palabras. Se generará un CSV vacío.")
        rows_to_csv([], [], csv_path)
        return

    processor, model = _load_layoutlmv3(model_id=model_id)
    log.info("Modelo LayoutLMv3 cargado.")

    labels = _predict_labels(processor, model, image, words, boxes)
    if len(labels) != len(words):
        log.warning(
            "Número de etiquetas (%d) no coincide con palabras (%d).",
            len(labels),
            len(words),
        )

    predictions = _build_predictions(words, boxes, labels)

    # Construir tokens y aplicar heurísticas estilo layout 'generic'
    tokens_generic: List[Token] = []
    for pred in predictions:
        token = Token(
            text=pred.text,
            page=1,
            x1=pred.bbox[0],
            y1=pred.bbox[1],
            x2=pred.bbox[2],
            y2=pred.bbox[3],
        )
        pred.token_ref = token
        tokens_generic.append(token)

    target_cols = expected_n_cols or max_columns or MAX_MODEL_COLUMNS
    try:
        lines = build_lines(tokens_generic)
        if lines:
            intervals = estimate_columns(
                lines,
                expected_n_cols=target_cols,
            )
            if intervals:
                records = assign_words_to_columns(lines, intervals)
                grid_rows = merge_lines_into_rows(records, lines)
                if grid_rows:
                    header_row, body_rows = detect_header_row(
                        grid_rows,
                        header_regexes=list(header_regexes) if header_regexes else None,
                    )
                    header = header_row or []
                    cleaned_rows = process_grid_data(body_rows)
                    rows_to_csv(cleaned_rows, header, csv_path)
                    log.info("CSV reconstruido con heurística estilo 'generic' escrito en: %s", csv_path)
                    return
                else:
                    log.debug("merge_lines_into_rows produjo 0 filas; se intentará fallback LayoutLM puro.")
            else:
                log.debug("estimate_columns no detectó intervalos; se intentará fallback LayoutLM puro.")
        else:
            log.debug("No se pudieron agrupar líneas; se intentará fallback LayoutLM puro.")
    except Exception:
        log.exception("Fallo en la reconstrucción estilo 'generic'. Se intentará fallback LayoutLM puro.")

    # Fallback basado en únicamente las etiquetas del modelo
    header, rows = _compose_table_from_predictions(
        predictions,
        max_columns=target_cols or MAX_MODEL_COLUMNS,
    )

    if not rows:
        log.warning("LayoutLM no produjo filas útiles. Se usará fallback de grid espacial.")
        spatial_words = [
            SpatialWord(text=w, bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
            for w, box in zip(words, boxes)
        ]
        grid = build_grid_from_words(spatial_words)
        cleaned_rows = process_grid_data(grid.rows)
        rows_to_csv(cleaned_rows, grid.header, csv_path)
        return

    cleaned_rows = process_grid_data(rows)
    rows_to_csv(cleaned_rows, header, csv_path)
    log.info("CSV reconstruido con fallback LayoutLM escrito en: %s", csv_path)
