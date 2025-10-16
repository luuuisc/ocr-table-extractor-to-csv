# src/hocr_table_extractor/assign_financial.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

# Requiere al menos un dígito O bien un guion solitario (cero).
# Soporta $, comas, espacios de miles, decimales y negativos con paréntesis.
NUM_TOKEN_RE = re.compile(r"""^(
    -                                   # guion solo (lo trataremos como cero)
    |
    \$?\(?-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?  # $ 1,234.56 o (57,519) o 246
)$""", re.VERBOSE)

def _is_numeric_span(txt: str) -> bool:
    s = txt.strip()
    # eliminar espacios internos redundantes para tokens como "$ 101,606"
    s = s.replace(" ", "")
    return bool(NUM_TOKEN_RE.match(s))

def _merge_adjacent_tokens(tokens: List[Any], max_gap_px: int = 18) -> List[Tuple[int, int, str]]:
    """Une tokens contiguos horizontalmente → spans [(x1,x2,text), ...]."""
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: t.x1)
    spans: List[Tuple[int, int, str]] = []
    cur_x1, cur_x2 = toks[0].x1, toks[0].x2
    cur_text = [toks[0].text]

    for t in toks[1:]:
        gap = t.x1 - cur_x2
        if gap <= max_gap_px:
            cur_text.append(t.text)
            cur_x2 = max(cur_x2, t.x2)
        else:
            spans.append((cur_x1, cur_x2, " ".join(cur_text).strip()))
            cur_x1, cur_x2 = t.x1, t.x2
            cur_text = [t.text]
    spans.append((cur_x1, cur_x2, " ".join(cur_text).strip()))
    return spans

def assign_financial_three_columns(lines: List[Dict[str, Any]],
                                   label_col_name: str = "Cuenta",
                                   newest_on_right: bool = True
                                   ) -> List[Dict[str, Any]]:
    """
    Para cada línea:
      - fusiona tokens adyacentes (spans)
      - detecta spans numéricos (requiere dígito o '-' como cero)
      - toma los 2 numéricos más a la derecha como columnas de año
      - el resto a la izquierda se concatena como 'Cuenta'

    Devuelve dict por línea:
      {
        page, y_top, y_bot,
        cells=[Cuenta, col_A, col_B],
        meta={"num_count": int, "has_label": bool}
      }
    """
    records = []
    for ln in lines:
        y1, y2 = ln["bbox"][1], ln["bbox"][3]
        spans = _merge_adjacent_tokens(ln["tokens"], max_gap_px=18)

        if not spans:
            records.append(dict(page=ln["page"], y_top=y1, y_bot=y2,
                                cells=["", "", ""], meta={"num_count": 0, "has_label": False}))
            continue

        numeric_spans = [(x1, x2, txt) for (x1, x2, txt) in spans if _is_numeric_span(txt)]
        text_spans    = [(x1, x2, txt) for (x1, x2, txt) in spans if not _is_numeric_span(txt)]
        num_sorted = sorted(numeric_spans, key=lambda s: s[0])

        col_A = col_B = ""
        if len(num_sorted) >= 2:
            rightmost = num_sorted[-1][2]
            second_right = num_sorted[-2][2]
            if newest_on_right:
                col_A, col_B = second_right, rightmost
            else:
                col_A, col_B = rightmost, second_right
        elif len(num_sorted) == 1:
            col_A = num_sorted[0][2]

        label_parts = [txt for (_, _, txt) in sorted(text_spans, key=lambda s: s[0])]
        label = " ".join(label_parts).strip()
        has_label = bool(label)

        records.append(dict(
            page=ln["page"], y_top=y1, y_bot=y2,
            cells=[label, col_A, col_B],
            meta={"num_count": len(num_sorted), "has_label": has_label}
        ))
    return records
