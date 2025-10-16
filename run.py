# run.py
from __future__ import annotations
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent / "src"))
from hocr_table_extractor.main import hocr_to_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def main() -> None:
    hocr_path = Path("/Users/luuiscc_/Documents/LUIS_DEVELOPER/jr LC/OCR_project/1C.hocr")
    csv_path  = Path("/Users/luuiscc_/Documents/LUIS_DEVELOPER/jr LC/OCR_project/salida.csv")

    table_bbox = None  # si ves ruido externo, define (x1,y1,x2,y2)

    logging.info(f"HOCR: {hocr_path}")
    logging.info(f"CSV : {csv_path}")

    try:
        hocr_to_csv(
            str(hocr_path),
            str(csv_path),
            table_bbox=table_bbox,
            layout="financial",     # << clave
            row_merge_factor=1.30,  # probemos conservador; si faltan celdas, subimos a 1.40–1.50
        )
        logging.info("✔ Proceso completado.")
    except Exception as e:
        logging.exception(f"❌ Error al procesar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
