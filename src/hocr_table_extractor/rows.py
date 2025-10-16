# src/hocr_table_extractor/rows.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from statistics import median
import re

def merge_lines_into_rows(records: List[Dict[str, Any]],
                          row_merge_factor: float = 1.25
                          ) -> List[List[str]]:
    """
    Merge genérico por proximidad vertical.
    Une líneas consecutivas si el gap vertical <= factor * mediana_altura_de_línea.
    Concatena texto por columna (preferencia simple).
    """
    if not records:
        return []

    heights = [r["y_bot"] - r["y_top"] for r in records]
    h_med = median(heights) if heights else 12
    max_gap = int(row_merge_factor * h_med)

    rows: List[List[str]] = []
    cur = records[0]["cells"][:]
    prev_bot = records[0]["y_bot"]

    for r in records[1:]:
        gap = r["y_top"] - prev_bot
        if gap <= max_gap:
            # concatenar celda a celda
            cur = [
                (" ".join([a, b]).strip() if a and b else (a or b))
                for a, b in zip(cur, r["cells"])
            ]
            prev_bot = max(prev_bot, r["y_bot"])
        else:
            rows.append(cur)
            cur = r["cells"][:]
            prev_bot = r["y_bot"]

    rows.append(cur)
    return rows


def merge_financial_rows(records: List[Dict[str, Any]],
                         row_merge_factor: float = 1.30
                         ) -> List[List[str]]:
    """
    Merge 'inteligente' para estados financieros (3 columnas: Cuenta | AñoA | AñoB).

    Reglas:
    - Solo une si la segunda línea NO trae importes (wrap de etiqueta), o
    - Si la primera NO trae importes y la segunda SÍ (etiqueta en línea 1, valores en 2).
    - Nunca une dos líneas que ambas traen importes (evita colapsar filas reales).

    Se asume que cada record tiene:
      - "cells": List[str] con 3 columnas [label, col_A, col_B]
      - "meta": Dict con "num_count" (int), opcional (default 0)
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
            # 1) wrap de etiqueta (siguiente sin importes)
            if r_num == 0:
                should_merge = True
            # 2) etiqueta en 1ra línea (sin importes) y valores en 2da
            elif cur_num == 0 and r_num > 0:
                should_merge = True

        if should_merge:
            # Fusiona: para la columna de 'Cuenta' concatenar; para numéricas, priorizar no vacíos
            merged: List[str] = []
            for idx, (a, b) in enumerate(zip(cur, r["cells"])):
                if idx == 0:
                    # columna label
                    merged.append(" ".join([a, b]).strip() if a and b else (a or b))
                else:
                    # columnas de importes: si ya hay valor en a, conservar; si no, tomar b
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
    Detección simple de cabecera:
    - Si se pasan regex, intenta encontrar una fila que las cumpla (en primeras 3).
    - Si no, toma la primera fila como encabezado por defecto.

    Devuelve (header, body). Si no hay filas, (None, []).
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

    # fallback: primera como header
    return candidate, rows[1:]
