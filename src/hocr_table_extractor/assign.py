from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

def assign_words_to_columns(lines: List[Dict[str, Any]],
                            columns: List[Tuple[int,int]]
                            ) -> List[Dict[str, Any]]:
    """Assign each token in a line to the closest column by horizontal center."""
    records = []
    if not columns:
        return records

    for ln in lines:
        y1, y2 = ln["bbox"][1], ln["bbox"][3]
        cells: List[List[str]] = [[] for _ in columns]
        for t in ln["tokens"]:
            xc = (t.x1 + t.x2) / 2.0
            idx = None
            for i, (L, R) in enumerate(columns):
                if L <= xc <= R:
                    idx = i; break
            if idx is None:
                dists = [min(abs(xc - L), abs(xc - R)) for (L, R) in columns]
                idx = int(np.argmin(dists))
            cells[idx].append(t.text)
        cells_joined = [" ".join(c).strip() for c in cells]
        records.append(dict(page=ln["page"], y_top=y1, y_bot=y2, cells=cells_joined))
    return records
