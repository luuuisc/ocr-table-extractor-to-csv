# run.py
from __future__ import annotations
import sys
from pathlib import Path
import logging
import argparse

sys.path.append(str(Path(__file__).parent / "src"))
from hocr_table_extractor.main import hocr_to_csv

# La configuración del logging se hará en main.py, pero dejamos un logger básico aquí.
log = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extraer tablas de archivos hOCR a CSV.")
    parser.add_argument("hocr_path", type=str, help="Ruta al archivo de entrada .hocr")
    parser.add_argument("csv_path", type=str, help="Ruta al archivo de salida .csv")
    parser.add_argument("--layout", type=str, default="dynamic", choices=["financial", "dynamic", "generic"],
                        help="Modo de extracción de tabla (default: dynamic)")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="Bbox opcional de la tabla: x1 y1 x2 y2")
    parser.add_argument("--expected-n-cols", type=int, help="Número esperado de columnas (para layout 'generic')")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nivel de verbosidad del log (default: INFO)")

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    log.info(f"HOCR: {args.hocr_path}")
    log.info(f"CSV : {args.csv_path}")

    try:
        hocr_to_csv(
            hocr_path=args.hocr_path,
            csv_path=args.csv_path,
            layout=args.layout,
            table_bbox=tuple(args.bbox) if args.bbox else None,
            expected_n_cols=args.expected_n_cols
        )
        log.info("✔ Proceso completado.")
    except FileNotFoundError:
        log.error(f"Error: No se encontró el archivo de entrada: {args.hocr_path}")
    except Exception as e:
        log.error(f"Ocurrió un error inesperado: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()