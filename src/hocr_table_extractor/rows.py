# src/hocr_table_extractor/rows.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from statistics import median
import re
import numpy as np

def merge_lines_into_rows(records: List[Dict[str, Any]],
                          lines: List[Dict[str, Any]],
                          ) -> List[List[str]]:
    """Fusiona registros en filas utilizando perfiles de proyección horizontal.

    Este método es más robusto que basarse en la altura media de las líneas.
    """
    if not records:
        return []

    all_tokens = [tok for ln in lines for tok in ln["tokens"] if tok.text]
    if not all_tokens:
        return [r["cells"] for r in records]

    y_min = min(t.y1 for t in all_tokens)
    y_max = max(t.y2 for t in all_tokens)

    # Crear un perfil de proyección horizontal
    profile = np.zeros(y_max - y_min, dtype=int)
    for token in all_tokens:
        start = token.y1 - y_min
        end = token.y2 - y_min
        profile[start:end] += 1

    # Encontrar los valles (gaps) en el perfil
    zero_indices = np.where(profile == 0)[0]
    if len(zero_indices) == 0:
        # No hay gaps, probablemente es una sola fila gigante
        final_row = ["" for _ in records[0]["cells"]]
        for r in records:
            final_row = [(" ".join([a, b]).strip() if a and b else (a or b)) for a, b in zip(final_row, r["cells"])]
        return [final_row]

    gaps = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    cuts = [y_min]
    for gap in gaps:
        if len(gap) > 2:  # Umbral mínimo para un separador de fila
            cuts.append(y_min + int(gap.mean()))
    cuts.append(y_max)
    cuts = sorted(list(set(cuts)))

    row_intervals = []
    for t, b in zip(cuts, cuts[1:]):
        if b - t > 5: # Umbral de altura mínima de fila
            row_intervals.append((t, b))

    # Asignar cada registro a un intervalo de fila
    rows_map = {i: [] for i in range(len(row_intervals))}
    for record in records:
        yc = (record["y_top"] + record["y_bot"]) / 2
        for i, (top, bot) in enumerate(row_intervals):
            if top <= yc < bot:
                rows_map[i].append(record["cells"])
                break

    # Fusionar los registros dentro de cada fila
    final_rows = []
    for i in sorted(rows_map.keys()):
        if not rows_map[i]:
            continue
        
        # Asegurarse de que todas las celdas tengan la misma longitud
        max_len = max(len(cells) for cells in rows_map[i])
        for cells in rows_map[i]:
            while len(cells) < max_len:
                cells.append("")

        final_row = ["" for _ in range(max_len)]
        for cells in rows_map[i]:
            final_row = [(" ".join([a, b]).strip() if a and b else (a or b)) for a, b in zip(final_row, cells)]
        final_rows.append(final_row)

    return final_rows


def merge_financial_rows(records: List[Dict[str, Any]],
                         row_merge_factor: float = 1.30
                         ) -> List[List[str]]:
    """
    Merge 'inteligente' para balances (Cuenta | AñoA | AñoB).

    Reglas de unión ENTRE LÍNEAS ADYACENTES:
      - Une si la segunda NO trae importes (wrap de etiqueta), o
      - Une si la primera NO trae importes y la segunda SÍ (etiqueta→valores).
      - NUNCA une dos líneas que ambas traen importes (para no colapsar filas reales).
    """
    if not records:
        return []

    heights = [r["y_bot"] - r["y_top"] for r in records]
    h_med = median(heights) if heights else 12
    max_gap = int(row_merge_factor * h_med)

    rows: List[List[str]] = []
    cur = records[0]["cells"][:]
    cur_num = int(records[0].get("meta", {}).get("num_count", 0))
    prev_bot = records[0]["y_bot"]

    for r in records[1:]:
        gap = r["y_top"] - prev_bot
        r_num = int(r.get("meta", {}).get("num_count", 0))

        should_merge = False
        if gap <= max_gap:
            if r_num == 0:                    # wrap puro de etiqueta
                should_merge = True
            elif cur_num == 0 and r_num > 0:  # etiqueta en 1ra y valores en 2da
                should_merge = True

        if should_merge:
            merged: List[str] = []
            for idx, (a, b) in enumerate(zip(cur, r["cells"])):
                if idx == 0:  # etiqueta
                    merged.append(" ".join([a, b]).strip() if a and b else (a or b))
                else:         # numéricas
                    merged.append(a if a else b)
            cur = merged
            cur_num = max(cur_num, r_num)
            prev_bot = max(prev_bot, r["y_bot"])
        else:
            rows.append(cur)
            cur = r["cells"][:]
            cur_num = r_num
            prev_bot = r["y_bot"]

    rows.append(cur)
    return rows


def detect_header_row(rows: List[List[str]],
                      header_regexes: Optional[List[str]] = None
                      ) -> Tuple[Optional[List[str]], List[List[str]]]:
    """
    Detección simple de cabecera para el modo 'generic'.
    """
    if not rows:
        return None, []

    candidate = rows[0]
    if header_regexes:
        patt = [re.compile(rx) for rx in header_regexes]
        def _match(row: List[str]) -> bool:
            text = " | ".join((c or "").lower() for c in row)
            return any(p.search(text) for p in patt)

        if _match(candidate):
            return candidate, rows[1:]

        for i in range(1, min(3, len(rows))):
            if _match(rows[i]):
                hdr = rows[i]
                body = rows[:i] + rows[i+1:]
                return hdr, body

    return candidate, rows[1:]
