from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

def estimate_columns(lines: List[Dict[str, Any]],
                     min_col_width: int = 25,
                     expected_n_cols: Optional[int] = None
                     ) -> List[Tuple[int,int]]:
    """Estima los intervalos de las columnas usando un perfil de proyección vertical.

    Este método es más robusto que basarse en los gaps entre palabras, ya que es
    menos sensible a la alineación y al espaciado irregular.
    """
    all_tokens = [tok for ln in lines for tok in ln["tokens"] if tok.text]
    if not all_tokens:
        return []

    x_min = min(t.x1 for t in all_tokens)
    x_max = max(t.x2 for t in all_tokens)

    # Crear un perfil de proyección vertical (un histograma horizontal)
    profile = np.zeros(x_max - x_min, dtype=int)
    for token in all_tokens:
        start = token.x1 - x_min
        end = token.x2 - x_min
        profile[start:end] += 1

    # Encontrar los valles (gaps) en el perfil
    zero_indices = np.where(profile == 0)[0]
    if len(zero_indices) == 0:
        # No hay gaps, probablemente es una sola columna
        return [(x_min, x_max)]

    # Agrupar índices de ceros consecutivos para encontrar los centros de los valles
    gaps = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    cuts = [x_min]
    for gap in gaps:
        if len(gap) > 5:  # Umbral mínimo para considerar un gap como separador
            cut_point = x_min + int(gap.mean())
            cuts.append(cut_point)
    cuts.append(x_max)
    cuts = sorted(list(set(cuts)))

    # Crear intervalos a partir de los puntos de corte
    intervals: List[Tuple[int,int]] = []
    for l, r in zip(cuts, cuts[1:]):
        if r - l >= min_col_width:
            intervals.append((l, r))

    # Forzar el número de columnas si se especifica
    if expected_n_cols and expected_n_cols > 0 and len(intervals) != expected_n_cols:
        # Si se detectan más columnas, fusionar las más cercanas
        while len(intervals) > expected_n_cols:
            dists = [intervals[i+1][0] - intervals[i][1] for i in range(len(intervals)-1)]
            if not dists:
                break
            j = int(np.argmin(dists))
            merged = (intervals[j][0], intervals[j+1][1])
            intervals = intervals[:j] + [merged] + intervals[j+2:]

        # Si se detectan menos columnas, dividir la más ancha
        while len(intervals) < expected_n_cols:
            widths = [iv[1] - iv[0] for iv in intervals]
            if not widths:
                break
            j = int(np.argmax(widths))
            wide_interval = intervals[j]
            # Dividir por la mitad
            mid_point = wide_interval[0] + widths[j] // 2
            intervals = intervals[:j] + [(wide_interval[0], mid_point), (mid_point, wide_interval[1])] + intervals[j+1:]

    return intervals
