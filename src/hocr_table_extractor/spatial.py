# src/hocr_table_extractor/spatial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class BBox:
    """Representa un bounding box con coordenadas x1, y1, x2, y2."""
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass(frozen=True)
class SpatialWord:
    """Representa una palabra con su texto y su posición espacial (bbox)."""
    text: str
    bbox: BBox

@dataclass
class TableGrid:
    """Representa la tabla como una rejilla de celdas."""
    rows: List[List[str]]
    header: List[str]
