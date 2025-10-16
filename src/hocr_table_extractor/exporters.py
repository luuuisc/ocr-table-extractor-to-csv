# src/hocr_table_extractor/exporters.py
from __future__ import annotations
from typing import List
import csv
import re

def rows_to_csv(rows: List[List[str]], header: List[str], csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)

_number_re = re.compile(r"[^\d\-\.\)]")  # quita $, comas, espacios

def _to_number(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    if not s: return ""
    if s == "-": return "0"
    neg = (s.startswith("(") and s.endswith(")"))
    s2 = _number_re.sub("", s)
    if not s2: return ""
    # manejar (57,519) → -57519 ya quitamos comas con regex
    try:
        val = float(s2)
        if neg:
            val = -val
        # exporta int si es entero
        if abs(val - int(val)) < 1e-9:
            return str(int(val))
        return str(val)
    except Exception:
        return s  # último recurso: deja el texto

def rows_to_csv_numeric(rows: List[List[str]], header: List[str], csv_path: str) -> None:
    """
    Igual que rows_to_csv pero convierte columnas numéricas a números en texto.
    Asume que col 0 es label y el resto numéricas.
    """
    norm_rows = []
    for r in rows:
        if not r:
            norm_rows.append(r); continue
        label = r[0]
        nums = [_to_number(x) for x in r[1:]]
        norm_rows.append([label] + nums)
    rows_to_csv(norm_rows, header, csv_path)
