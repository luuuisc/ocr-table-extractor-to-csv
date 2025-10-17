# src/hocr_table_extractor/cleaners.py
from __future__ import annotations
import logging
from typing import List

log = logging.getLogger(__name__)

def clean_cell_text(text: str) -> str:
    """Limpia el texto de una celda individual."""
    # Lógica para remover artefactos, etc.
    return text.strip()

def process_grid_data(grid: List[List[str]]) -> List[List[str]]:
    """Aplica funciones de limpieza a toda la rejilla."""
    log.info("Procesando y limpiando datos de la rejilla.")
    
    processed_grid = []
    for row in grid:
        cleaned_row = [clean_cell_text(cell) for cell in row]
        processed_grid.append(cleaned_row)
        
    # Aquí se podría añadir lógica para:
    # - Convertir números
    # - Detectar jerarquías
    
    return processed_grid
