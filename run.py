# run.py
from __future__ import annotations
import sys
from pathlib import Path
import logging
import argparse

sys.path.append(str(Path(__file__).parent / "src"))
from importlib import import_module
hocr_main = import_module("hocr_table_extractor.main")
hocr_to_csv = hocr_main.hocr_to_csv

# La configuración del logging se hará en main.py, pero dejamos un logger básico aquí.
log = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extraer tablas de archivos hOCR o imágenes a CSV.")
    parser.add_argument("csv_path", type=str, help="Ruta al archivo de salida .csv")
    parser.add_argument("--hocr_path", type=str, help="Ruta al archivo de entrada .hocr (para layouts tradicionales)")
    parser.add_argument("--image", type=str, help="Ruta a la imagen de entrada (para layout 'transformers')")
    parser.add_argument("--layout", type=str, default="dynamic", choices=["financial", "dynamic", "generic", "professional", "transformers"],
                        help="Modo de extracción de tabla (default: dynamic)")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="Bbox opcional de la tabla: x1 y1 x2 y2")
    parser.add_argument("--expected-n-cols", type=int, help="Número esperado de columnas (para layout 'generic')")
    parser.add_argument("--transformer-model", type=str, help="Checkpoint de LayoutLM para el layout 'transformers'.")
    parser.add_argument("--transformer-ocr-lang", type=str, help="Idioma OCR para Tesseract (layout 'transformers').")
    parser.add_argument("--transformer-max-cols", type=int, help="Máximo de columnas a reconstruir con LayoutLM.")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nivel de verbosidad del log (default: INFO)")

    args = parser.parse_args()

    # --- Validación de argumentos ---
    if args.layout == "transformers":
        if not args.image:
            parser.error("--image es requerido para --layout 'transformers'")
        log.info(f"IMAGEN: {args.image}")
    else:
        if not args.hocr_path:
            parser.error("--hocr_path es requerido para el layout '{}'".format(args.layout))
        log.info(f"HOCR: {args.hocr_path}")

    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info(f"CSV : {args.csv_path}")

    try:
        hocr_to_csv(
            hocr_path=args.hocr_path,
            csv_path=args.csv_path,
            image_path=args.image,
            layout=args.layout,
            table_bbox=tuple(args.bbox) if args.bbox else None,
            expected_n_cols=args.expected_n_cols,
            transformer_model=args.transformer_model,
            transformer_ocr_lang=args.transformer_ocr_lang,
            transformer_max_columns=args.transformer_max_cols,
        )
        log.info("✔ Proceso completado.")
    except FileNotFoundError:
        log.error(f"Error: No se encontró el archivo de entrada: {args.hocr_path or args.image}")
    except Exception as e:
        log.error(f"Ocurrió un error inesperado: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
