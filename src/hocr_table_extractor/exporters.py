from __future__ import annotations
from typing import List
import pandas as pd

def rows_to_csv(rows: List[List[str]], header: List[str], csv_path: str) -> None:
    n = len(header)
    body_norm = [r + [""]*(n - len(r)) if len(r) < n else r[:n] for r in rows]
    df = pd.DataFrame(body_norm, columns=[h if h else f"col_{i+1}" for i, h in enumerate(header)])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
