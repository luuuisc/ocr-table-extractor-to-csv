from __future__ import annotations
from typing import Any, Dict, List, Tuple
from statistics import median
from .structures import Token, overlap_ratio

def build_lines(tokens: List[Token]) -> List[Dict[str, Any]]:
    """Group tokens into lines. If line_id exists, use it; otherwise infer.

    Returns a list of dicts: {page, line_id, bbox, tokens}
    """
    if not tokens:
        return []

    if any(t.line_id for t in tokens):
        lines: Dict[Tuple[int,str], List[Token]] = {}
        for t in tokens:
            lid = t.line_id or f"inferred_{t.page}_{int(t.yc)}"
            lines.setdefault((t.page, lid), []).append(t)
        result = []
        for (page, lid), toks in lines.items():
            toks.sort(key=lambda z: z.x1)
            x1 = min(t.x1 for t in toks); y1 = min(t.y1 for t in toks)
            x2 = max(t.x2 for t in toks); y2 = max(t.y2 for t in toks)
            result.append(dict(page=page, line_id=lid, bbox=(x1,y1,x2,y2), tokens=toks))
        result.sort(key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))
        return result

    # Infer lines by vertical overlap
    tokens_sorted = sorted(tokens, key=lambda t: (t.page, t.yc, t.x1))
    lines: List[Dict[str, Any]] = []
    current: List[Token] = []
    current_page = tokens_sorted[0].page
    y_band: Tuple[int,int] = (tokens_sorted[0].y1, tokens_sorted[0].y2)

    for tok in tokens_sorted:
        if tok.page != current_page:
            if current:
                x1 = min(t.x1 for t in current); y1 = min(t.y1 for t in current)
                x2 = max(t.x2 for t in current); y2 = max(t.y2 for t in current)
                lines.append(dict(page=current_page, line_id=None, bbox=(x1,y1,x2,y2), tokens=sorted(current, key=lambda z: z.x1)))
            current = [tok]
            current_page = tok.page
            y_band = (tok.y1, tok.y2)
            continue

        if overlap_ratio(y_band[0], y_band[1], tok.y1, tok.y2) >= 0.5:
            current.append(tok)
            y_band = (min(y_band[0], tok.y1), max(y_band[1], tok.y2))
        else:
            if current:
                x1 = min(t.x1 for t in current); y1 = min(t.y1 for t in current)
                x2 = max(t.x2 for t in current); y2 = max(t.y2 for t in current)
                lines.append(dict(page=current_page, line_id=None, bbox=(x1,y1,x2,y2), tokens=sorted(current, key=lambda z: z.x1)))
            current = [tok]
            y_band = (tok.y1, tok.y2)

    if current:
        x1 = min(t.x1 for t in current); y1 = min(t.y1 for t in current)
        x2 = max(t.x2 for t in current); y2 = max(t.y2 for t in current)
        lines.append(dict(page=current_page, line_id=None, bbox=(x1,y1,x2,y2), tokens=sorted(current, key=lambda z: z.x1)))

    lines.sort(key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))
    return lines
