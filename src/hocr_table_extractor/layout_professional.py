# src/hocr_table_extractor/layout_professional.py
from __future__ import annotations
import logging
from typing import Optional, Tuple

from .parser import parse_hocr_words
from .spatial import BBox, SpatialWord
from .grid_builder import build_grid_from_words
from .cleaners import process_grid_data
from .exporters import rows_to_csv

log = logging.getLogger(__name__)

def extract_professional_layout(
    hocr_path: str,
    csv_path: str,
    table_bbox: Optional[Tuple[int, int, int, int]] = None
) -> None:
    """
    Pipeline de extracción profesional basada en análisis espacial.
    """
    log.info("=== Iniciando layout 'professional' ===")

    # --- PASO 1: Parseo a objetos espaciales ---
    log.info(f"Paso 1: Parseando HOCR '{hocr_path}' a objetos espaciales...")
    raw_words = parse_hocr_words(hocr_path, table_bbox=table_bbox)
    if not raw_words:
        log.warning("No se encontraron palabras en el HOCR. Se generará un CSV vacío.")
        rows_to_csv([], [], csv_path)
        return

    words: List[SpatialWord] = []
    for w in raw_words:
        try:
            box = BBox(x1=w.x1, y1=w.y1, x2=w.x2, y2=w.y2)
            words.append(SpatialWord(text=w.text, bbox=box))
        except AttributeError:
            log.warning(f"El token '{w}' no tiene atributos de coordenadas. Se omitirá.")

    log.info(f"Se crearon {len(words)} objetos SpatialWord.")

    # --- PASO 2: Construcción de la rejilla (grid) ---
    # (Aquí llamaremos a grid_builder.py)
    log.info("Paso 2: Construyendo la rejilla de la tabla...")
    grid = build_grid_from_words(words)

    # --- PASO 3: Limpieza y reconstrucción de datos ---
    # (Aquí llamaremos a cleaners.py)
    log.info("Paso 3: Limpiando y procesando datos de celdas...")
    cleaned_rows = process_grid_data(grid.rows)

    # --- PASO 4: Exportación a CSV ---
    # (Aquí llamaremos a exporters.py)
    log.info("Paso 4: Exportando a CSV...")
    rows_to_csv(cleaned_rows, grid.header, csv_path)

    log.info(f"✔ Layout 'professional' completado. CSV guardado en: {csv_path}")
