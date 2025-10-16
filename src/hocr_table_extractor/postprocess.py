# src/hocr_table_extractor/postprocess.py
from __future__ import annotations
from typing import List, Tuple
import re

SECTION_RE = re.compile(r":\s*$")  # líneas de sección terminan en ":" (e.g., "Activo circulante:")
FOOTER_RE  = re.compile(r"las notas adjuntas", re.IGNORECASE)

def _is_number_like(s: str) -> bool:
    if not s:
        return False
    z = s.strip().replace(" ", "")
    if z == "-":
        return True
    # $1,234.56 / (57,519) / 246
    return bool(re.match(r"^\$?\(?-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?$", z))

def fill_missing_labels_and_clean(rows: List[List[str]],
                                  label_for_subtotals: bool = True,
                                  normalize_dash_zero: bool = True
                                  ) -> List[List[str]]:
    """
    Posprocesa filas 'financial':
      - Si label vacío y hay dos números → etiqueta como 'Total <última sección>'.
      - Elimina leyenda de pie de página.
      - Normaliza '-' a '0' (opcional).
    """
    clean: List[List[str]] = []
    last_section: str = ""

    for cells in rows:
        # sanitize longitud
        a, y1, y2 = (cells + ["", "", ""])[:3]
        text = (a or "").strip()

        # footer
        if FOOTER_RE.search(text):
            continue

        # actualizar sección si termina con ':'
        if SECTION_RE.search(text):
            last_section = text.rstrip(":").strip()
            # la fila de sección se conserva sin importes
            clean.append([text, "", ""])
            continue

        # etiquetar subtotales sin label
        if label_for_subtotals and not text and _is_number_like(y1) and _is_number_like(y2):
            label = f"Total {last_section}" if last_section else "Subtotal"
            a = label

        # normalizar guiones
        if normalize_dash_zero:
            if y1 and y1.strip() == "-":
                y1 = "0"
            if y2 and y2.strip() == "-":
                y2 = "0"

        clean.append([a, y1, y2])

    return clean
