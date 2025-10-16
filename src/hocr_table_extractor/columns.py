from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

def percentile(arr: List[float], q: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(np.array(arr, dtype=float), q))

def estimate_columns(lines: List[Dict[str, Any]],
                     col_gap_quantile: float = 92.0,
                     min_col_width: int = 25,
                     expected_n_cols: Optional[int] = None
                     ) -> List[Tuple[int,int]]:
    """Estimate column intervals from global inter-word gaps across lines."""
    all_xs: List[int] = []
    all_gaps: List[int] = []
    for ln in lines:
        toks = [t for t in ln["tokens"] if t.text]
        toks.sort(key=lambda z: z.x1)
        for a, b in zip(toks, toks[1:]):
            gap = b.x1 - a.x2
            if gap > 0:
                all_gaps.append(gap)
        for t in toks:
            all_xs.extend([t.x1, t.x2])

    if not all_xs:
        return []

    x_min, x_max = min(all_xs), max(all_xs)
    if not all_gaps:
        return [(x_min, x_max)]

    thr = max(5, int(percentile(all_gaps, col_gap_quantile)))
    cuts: List[int] = [x_min]
    for ln in lines:
        toks = [t for t in ln["tokens"] if t.text]
        toks.sort(key=lambda z: z.x1)
        for a, b in zip(toks, toks[1:]):
            gap = b.x1 - a.x2
            if gap >= thr:
                cuts.append((a.x2 + b.x1) // 2)
    cuts.append(x_max)
    cuts = sorted(set(cuts))

    intervals: List[Tuple[int,int]] = []
    for l, r in zip(cuts, cuts[1:]):
        if r - l >= min_col_width:
            intervals.append((l, r))

    if expected_n_cols and expected_n_cols > 0:
        if len(intervals) < expected_n_cols:
            width = x_max - x_min
            step = width // expected_n_cols
            intervals = [(x_min + i*step, x_min + (i+1)*step) for i in range(expected_n_cols)]
        elif len(intervals) > expected_n_cols:
            while len(intervals) > expected_n_cols:
                dists = [intervals[i+1][0] - intervals[i][1] for i in range(len(intervals)-1)]
                j = int(np.argmin(dists))
                merged = (intervals[j][0], intervals[j+1][1])
                intervals = intervals[:j] + [merged] + intervals[j+2:]

    return intervals
