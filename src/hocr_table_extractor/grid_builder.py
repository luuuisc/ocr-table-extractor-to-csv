# src/hocr_table_extractor/grid_builder.py
from __future__ import annotations
import logging
from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np

from .spatial import SpatialWord, TableGrid
from .structures import Token
from .assign import assign_words_to_columns
from .rows import merge_lines_into_rows

log = logging.getLogger(__name__)

def group_words_into_lines(words: List[SpatialWord], tolerance: int = 5) -> List[List[SpatialWord]]:
    """Agrupa palabras en líneas basado en la superposición de sus coordenadas Y."""
    if not words:
        return []
    words.sort(key=lambda w: (w.bbox.y1, w.bbox.x1))
    lines = []
    current_line = [words[0]]
    for word in words[1:]:
        if abs(word.bbox.y1 - current_line[-1].bbox.y1) <= tolerance:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda w: w.bbox.x1))
            current_line = [word]
    lines.append(sorted(current_line, key=lambda w: w.bbox.x1))
    return lines

def estimate_column_positions(words: List[SpatialWord], min_col_width: int = 25, min_gap_width: int = 5) -> List[Tuple[int, int]]:
    """Estima los límites de las columnas usando un perfil de proyección vertical."""
    if not words:
        return []

    x_min = min(w.bbox.x1 for w in words)
    x_max = max(w.bbox.x2 for w in words)

    profile = np.zeros(x_max - x_min, dtype=int)
    for word in words:
        start = word.bbox.x1 - x_min
        end = word.bbox.x2 - x_min
        profile[start:end] += 1

    zero_indices = np.where(profile == 0)[0]
    if len(zero_indices) == 0:
        return [(x_min, x_max)]

    gaps = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    cuts = [x_min]
    for gap in gaps:
        if len(gap) > min_gap_width:
            cuts.append(x_min + int(gap.mean()))
    cuts.append(x_max)
    
    intervals = []
    if cuts:
        l = cuts[0]
        for r in cuts[1:]:
            if r - l > min_col_width:
                intervals.append((l, r))
                l = r
    return intervals

def find_header_row_index(rows: List[List], text_threshold: float = 0.7) -> Optional[int]:
    """Encuentra el índice de la fila de cabecera más probable."""
    best_candidate = -1
    max_text_ratio = 0.0
    for i, row in enumerate(rows):
        if not row or not row[0]: # Skip empty rows or rows with no description
            continue
        # Consider columns after the first for numeric check
        numeric_part = row[1:]
        if not numeric_part:
            continue
        text_cells = sum(1 for cell in numeric_part if isinstance(cell, str))
        text_ratio = text_cells / len(numeric_part)
        if text_ratio >= text_threshold and text_ratio > max_text_ratio:
            max_text_ratio = text_ratio
            best_candidate = i
    return best_candidate

def build_hierarchy(lines: List[List[SpatialWord]], grid_rows: List[List[str]]) -> Tuple[List[List[str]], int]:
    """Analiza la indentación para añadir columnas de jerarquía."""
    if not lines or not grid_rows:
        return grid_rows, 0

    indent_levels = [(line[0].bbox.x1, i) for i, line in enumerate(lines) if line]
    if not indent_levels:
        return grid_rows, 0

    # Simplistic hierarchy based on indentation
    parent_stack = [] # Stack of (indent, description)
    hierarchical_rows = []
    max_depth = 0

    for i, row in enumerate(grid_rows):
        current_indent = lines[i][0].bbox.x1 if lines[i] else -1
        description = row[0] if row else ""

        while parent_stack and current_indent <= parent_stack[-1][0]:
            parent_stack.pop()

        new_row = [p[1] for p in parent_stack] + row
        hierarchical_rows.append(new_row)
        max_depth = max(max_depth, len(parent_stack))

        # Add current item to stack if it looks like a parent
        # Heuristic: A parent has content in the first column and maybe not much else
        is_potential_parent = description and (len(row) < 3 or all(c == '' for c in row[1:]))
        if is_potential_parent:
            parent_stack.append((current_indent, description))

    return hierarchical_rows, max_depth

def build_grid_from_words(words: List[SpatialWord]) -> TableGrid:
    """
    Toma una lista de SpatialWords y las organiza en una rejilla (TableGrid).
    """
    log.info("Iniciando la construcción de la rejilla espacial...")
    if not words:
        return TableGrid(rows=[], header=[])

    lines = group_words_into_lines(words)
    log.info(f"Se agruparon las palabras en {len(lines)} líneas.")

    column_intervals = estimate_column_positions(words)
    log.info(f"Se estimaron {len(column_intervals)} columnas en las posiciones: {column_intervals}")

    grid_rows = []
    for line in lines:
        row = ["" for _ in column_intervals]
        for word in line:
            col_idx = -1
            word_center_x = (word.bbox.x1 + word.bbox.x2) / 2
            for i, interval in enumerate(column_intervals):
                if interval[0] <= word_center_x < interval[1]:
                    col_idx = i
                    break
            
            if col_idx != -1:
                row[col_idx] = (row[col_idx] + " " + word.text).strip()
        grid_rows.append(row)

    # --- Jerarquía ---
    hierarchical_rows, hierarchy_depth = build_hierarchy(lines, grid_rows)
    log.info(f"Profundidad de jerarquía detectada: {hierarchy_depth}")

    header_idx = find_header_row_index(hierarchical_rows)
    if header_idx is not None:
        log.info(f"Cabecera inteligente detectada en la fila {header_idx}.")
        header_row = hierarchical_rows[header_idx]
        body = hierarchical_rows[:header_idx] + hierarchical_rows[header_idx+1:]
        # Clean up hierarchy columns in header
        header = ["" for _ in range(hierarchy_depth)] + header_row[hierarchy_depth:]
    else:
        log.warning("No se encontró una cabecera clara. Se usará una cabecera genérica.")
        num_cols = len(hierarchical_rows[0]) if hierarchical_rows else 0
        header = [f"Level_{i+1}" for i in range(hierarchy_depth)] + \
                 [f"col_{i+1}" for i in range(num_cols - hierarchy_depth)]
        body = hierarchical_rows

    num_cols = len(header)
    for i in range(len(body)):
        body[i] = (body[i] + [""] * num_cols)[:num_cols]

    log.info(f"Rejilla construida con {len(body)} filas y {len(header)} columnas.")
    return TableGrid(rows=body, header=header)
