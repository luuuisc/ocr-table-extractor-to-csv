# src/hocr_table_extractor/assign_dynamic.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re
import numpy as np

NUM_RE = re.compile(r"""^
    [\$\(]?\s* -?
    (?:\d{1,3}(?:[,\s]\d{3})+|\d+)? (?:\.\d+)? \s*[\)]?
    $""", re.VERBOSE)

def _line_gap_quantile(tokens: List[Any], q: float = 95.0) -> int:
    gaps = []
    toks = sorted(tokens, key=lambda t: t.x1)
    for a, b in zip(toks, toks[1:]):
        g = b.x1 - a.x2
        if g > 0: gaps.append(g)
    if not gaps: return 18
    return max(12, int(np.percentile(np.array(gaps, dtype=float), q)))

def _merge_adjacent(tokens: List[Any], max_gap_px: int) -> List[Tuple[int,int,str]]:
    if not tokens: return []
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

def assign_dynamic(lines: List[Dict[str, Any]],
                   numeric_columns: List[Tuple[int,int]],
                   ) -> List[Dict[str, Any]]:
    records = []
    if not numeric_columns:
        for ln in lines:
            label = " ".join(t.text for t in sorted(ln["tokens"], key=lambda z: z.x1))
            records.append(dict(page=ln["page"], y_top=ln["bbox"][1], y_bot=ln["bbox"][3],
                                cells=[label], meta={"num_count":0}))
        return records

    cols = sorted(numeric_columns, key=lambda ab: ab[0])
    first_L = cols[0][0]

    for ln in lines:
        max_gap = _line_gap_quantile(ln["tokens"])
        spans = _merge_adjacent(ln["tokens"], max_gap_px=max_gap)

        num_spans = [(x1, x2, txt) for (x1, x2, txt) in spans if NUM_RE.match(txt.replace(" ", ""))]
        txt_spans = [(x1, x2, txt) for (x1, x2, txt) in spans if not NUM_RE.match(txt.replace(" ", ""))]

        label_parts = [txt for (x1, _, txt) in txt_spans if x1 < first_L]
        label = " ".join(label_parts).strip()

        values = [""] * len(cols)
        for (x1, x2, txt) in num_spans:
            xc = (x1 + x2) / 2.0
            dists = [0 if (L <= xc <= R) else min(abs(xc - L), abs(xc - R)) for (L, R) in cols]
            j = int(np.argmin(dists))
            values[j] = values[j] or txt.strip()

        records.append(dict(page=ln["page"], y_top=ln["bbox"][1], y_bot=ln["bbox"][3],
                            cells=[label] + values,
                            meta={"num_count": sum(1 for v in values if v)}))
    return records
