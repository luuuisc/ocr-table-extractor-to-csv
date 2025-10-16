# src/hocr_table_extractor/column_model.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
import numpy as np

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUM_RE = re.compile(r"""^
    [\$\(]?\s* -?
    (?:\d{1,3}(?:[,\s]\d{3})+|\d+)? (?:\.\d+)? \s*[\)]?
    $""", re.VERBOSE)

def _percentile(arr: List[float], q: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(np.asarray(arr, dtype=float), q))

def _line_gap_quantile(tokens: List[Any], q: float = 95.0) -> int:
    gaps = []
    toks = sorted(tokens, key=lambda t: t.x1)
    for a, b in zip(toks, toks[1:]):
        g = b.x1 - a.x2
        if g > 0:
            gaps.append(g)
    if not gaps:
        return 18
    return max(12, int(np.percentile(np.array(gaps, dtype=float), q)))

def _merge_adjacent(tokens: List[Any], max_gap_px: int) -> List[Tuple[int,int,str]]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: t.x1)
    spans = []
    x1, x2 = toks[0].x1, toks[0].x2
    buf = [toks[0].text]
    for t in toks[1:]:
        gap = t.x1 - x2
        if gap <= max_gap_px:
            buf.append(t.text)
            x2 = max(x2, t.x2)
        else:
            spans.append((x1, x2, " ".join(buf).strip()))
            x1, x2, buf = t.x1, t.x2, [t.text]
    spans.append((x1, x2, " ".join(buf).strip()))
    return spans

def _year_headers_from_top(lines: List[Dict[str, Any]],
                           intervals: List[Tuple[int,int]]) -> Optional[List[str]]:
    """Detecta años en la franja superior y los asigna al intervalo de X más cercano."""
    if not intervals:
        return None
    try:
        ys = [ln["bbox"][1] for ln in lines]
        if not ys:
            return None
        y_thr = min(ys) + 0.20 * (max(ys) - min(ys))
        candidates: List[Tuple[int, str]] = []
        for ln in lines:
            if ln["bbox"][1] <= y_thr:
                max_gap = _line_gap_quantile(ln["tokens"])
                spans = _merge_adjacent(ln["tokens"], max_gap_px=max_gap)
                for (x1, x2, txt) in spans:
                    m = YEAR_RE.search(txt)
                    if m:
                        xc = (x1 + x2) // 2
                        dists = [0 if (L <= xc <= R) else min(abs(xc - L), abs(xc - R)) for (L, R) in intervals]
                        j = int(np.argmin(dists))
                        candidates.append((j, m.group(0)))
        if not candidates:
            return None
        cols = len(intervals)
        col_names = [""] * cols
        for j in range(cols):
            ys = [yr for (idx, yr) in candidates if idx == j]
            if ys:
                vals, cnts = np.unique(np.array(ys), return_counts=True)
                col_names[j] = str(vals[int(np.argmax(cnts))])
        if any(col_names):
            return [nm if nm else f"Valor_{i+1}" for i, nm in enumerate(col_names)]
        return None
    except Exception:
        return None

def infer_numeric_columns_from_lines(lines: List[Dict[str, Any]],
                                     min_sep_px: int = 35,
                                     cut_quantile: float = 90.0,
                                     pad_px: int = 24
                                     ) -> Tuple[List[Tuple[int, int]], Optional[List[str]]]:
    """
    Estrategia híbrida (robusta):
    1) Por línea, fusiona spans y recoge centros X de NUMs.
       - Guarda por línea: rightmost, second-rightmost, third-rightmost (si existen).
       - Estima # de columnas por la **moda** del # de NUMs/line en el 70% inferior (evita headers).
       - Si moda >= 2: usa las **medianas** de esos picos por posición para definir centros de columnas.
         Construye intervalos tomando puntos medios entre centros y aplica pad_px.
    2) Si no hay suficiente señal, fallback al método clásico por **gaps globales** con padding.
    3) Intenta detectar **años** en encabezados y nombrar columnas.
    """
    if not lines:
        return [], None

    # --- Paso 1: spans y centros por línea ---
    per_line_centers: List[List[int]] = []
    y_vals = [ln["bbox"][1] for ln in lines]
    y_min, y_max = min(y_vals), max(y_vals)
    y_body_thr = y_min + 0.30 * (y_max - y_min)  # ignora 30% superior (headers) para la moda

    body_counts: List[int] = []

    for ln in lines:
        max_gap = _line_gap_quantile(ln["tokens"])
        spans = _merge_adjacent(ln["tokens"], max_gap_px=max_gap)
        centers = []
        for (x1, x2, txt) in spans:
            if NUM_RE.match(txt.replace(" ", "")):
                centers.append(int((x1 + x2) // 2))
        centers.sort()
        per_line_centers.append(centers)
        if ln["bbox"][1] >= y_body_thr:
            body_counts.append(len(centers))

    # moda del # de números por línea (cuerpo)
    ncols_guess = 0
    if body_counts:
        vals, cnts = np.unique(np.array(body_counts), return_counts=True)
        # ignorar líneas con 0 (títulos)
        mask = vals > 0
        if mask.any():
            vals2, cnts2 = vals[mask], cnts[mask]
            ncols_guess = int(vals2[int(np.argmax(cnts2))])

    # limitar a 4 columnas máximas
    if ncols_guess > 4:
        ncols_guess = 4

    intervals: List[Tuple[int,int]] = []

    if ncols_guess >= 2:
        # agregación por picos: rightmost, second-rightmost, third...
        buckets: List[List[int]] = [[] for _ in range(ncols_guess)]
        for centers in per_line_centers:
            if len(centers) >= 1:
                buckets[0].append(centers[-1])          # rightmost
            if len(centers) >= 2 and ncols_guess >= 2:
                buckets[1].append(centers[-2])          # second rightmost
            if len(centers) >= 3 and ncols_guess >= 3:
                buckets[2].append(centers[-3])
            if len(centers) >= 4 and ncols_guess >= 4:
                buckets[3].append(centers[-4])

        # si hay muy poca señal en alguna posición, volvemos al fallback
        if any(len(b) < max(5, 0.05 * len(per_line_centers)) for b in buckets):
            ncols_guess = 0  # forzar fallback
        else:
            centers_ordered = [int(np.median(b)) for b in buckets]  # robustez con mediana
            centers_ordered.sort()  # izq→der

            # construir límites por puntos medios + padding
            edges = []
            for a, b in zip(centers_ordered, centers_ordered[1:]):
                edges.append((a + b) // 2)
            # intervalos a partir de edges
            L = centers_ordered[0] - pad_px
            intervals = []
            for mid in edges:
                intervals.append((int(L), int(mid + pad_px)))
                L = int(mid - pad_px)
            intervals.append((int(L), int(centers_ordered[-1] + pad_px)))

    if not intervals:
        # --- Paso 2: fallback por gaps globales ---
        all_centers: List[int] = []
        for centers in per_line_centers:
            all_centers.extend(centers)
        if not all_centers:
            return [], None
        all_centers.sort()
        gaps = [b - a for a, b in zip(all_centers, all_centers[1:])]
        thr = max(min_sep_px, int(_percentile(gaps, cut_quantile)))
        cuts = [all_centers[0]]
        for a, b in zip(all_centers, all_centers[1:]):
            if (b - a) >= thr:
                cuts.append((a + b) // 2)
        cuts.append(all_centers[-1])

        raw_intervals: List[Tuple[int, int]] = []
        for L, R in zip(cuts, cuts[1:]):
            if R - L >= 10:
                raw_intervals.append((int(L), int(R)))

        merged: List[Tuple[int, int]] = []
        for iv in raw_intervals:
            if not merged or iv[0] - merged[-1][1] > 8:
                merged.append(iv)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], iv[1]))
        intervals = [(int(L - pad_px), int(R + pad_px)) for (L, R) in merged][:4]

    # --- Paso 3: intentar nombres con años ---
    names = _year_headers_from_top(lines, intervals) if intervals else None
    return intervals, names
