from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .dataset_builder import build_layoutlm_example
from .ocr_utils import generate_hocr_from_image

log = logging.getLogger(__name__)


def _parse_bbox(values: Sequence[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("El bbox debe tener exactamente 4 enteros: x1 y1 x2 y2")
    x1, y1, x2, y2 = map(int, values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("BBox inválido: x2 debe ser > x1 y y2 > y1")
    return x1, y1, x2, y2


def _normalize_extensions(exts: Iterable[str]) -> List[str]:
    return [ext if ext.startswith(".") else f".{ext}" for ext in exts]


def _discover_pairs_from_dir(
    image_dir: Path,
    *,
    hocr_dir: Path | None,
    extensions: List[str],
    auto_generate: bool,
    ocr_lang: str,
    ocr_psm: int,
    ocr_oem: int,
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(image_dir.glob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in extensions:
            continue

        if hocr_dir:
            candidate_hocr = hocr_dir / f"{img_path.stem}.hocr"
        else:
            candidate_hocr = img_path.with_suffix(".hocr")

        if candidate_hocr.exists():
            hocr_path = candidate_hocr
        elif auto_generate:
            hocr_path = Path(
                generate_hocr_from_image(
                    str(img_path),
                    str(candidate_hocr),
                    lang=ocr_lang,
                    psm=ocr_psm,
                    oem=ocr_oem,
                )
            )
        else:
            log.warning("No se encontró HOCR para %s. Se omite.", img_path)
            continue

        pairs.append((img_path, hocr_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un dataset JSONL para fine-tuning de LayoutLMv3-base."
    )
    parser.add_argument(
        "--pair",
        action="append",
        metavar="IMAGE:HOCR",
        help="Par de archivos separados por ':' (ej. imagen.jpeg:tabla.hocr). Puede repetirse.",
    )
    parser.add_argument(
        "--image-dir",
        help="Directorio con imágenes. Se buscará un .hocr con el mismo nombre; puede generarse con --generate-hocr.",
    )
    parser.add_argument(
        "--hocr-dir",
        help="Directorio alternativo donde buscar los HOCR correspondientes (mismos nombres base).",
    )
    parser.add_argument(
        "--generate-hocr",
        action="store_true",
        help="Genera HOCR automáticamente con Tesseract para las imágenes que no lo tengan.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"],
        help="Extensiones (con punto) a considerar al escanear --image-dir.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Idioma para Tesseract cuando se use --generate-hocr.",
    )
    parser.add_argument(
        "--ocr-psm",
        type=int,
        default=6,
        help="Valor --psm de Tesseract al generar HOCR (default: 6).",
    )
    parser.add_argument(
        "--ocr-oem",
        type=int,
        default=3,
        help="Valor --oem de Tesseract al generar HOCR (default: 3).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Archivo JSONL de salida.",
    )
    parser.add_argument(
        "--expected-n-cols",
        type=int,
        default=None,
        help="Número esperado de columnas para el maestro `generic`.",
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box opcional para recortar la tabla.",
    )
    parser.add_argument(
        "--header-regex",
        action="append",
        help="Expresiones regulares para detectar la fila de encabezado. Puede repetirse.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=6,
        help="Máximo de columnas etiquetadas explícitamente (resto se marca como OTHER).",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(levelname)s - %(message)s")

    bbox = _parse_bbox(args.bbox) if args.bbox else None
    extensions = _normalize_extensions(args.extensions)

    discovered_pairs: List[Tuple[Path, Path]] = []

    if args.pair:
        for pair in args.pair:
            try:
                image_path_str, hocr_path_str = pair.split(":", 1)
            except ValueError as exc:
                raise SystemExit(f"Formato inválido en --pair '{pair}'. Use IMAGE:HOCR") from exc
            image_path = Path(image_path_str).expanduser()
            hocr_path = Path(hocr_path_str).expanduser()
            discovered_pairs.append((image_path, hocr_path))

    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser()
        if not image_dir.is_dir():
            raise SystemExit(f"--image-dir no es un directorio válido: {image_dir}")
        hocr_dir = Path(args.hocr_dir).expanduser() if args.hocr_dir else None
        pairs_from_dir = _discover_pairs_from_dir(
            image_dir,
            hocr_dir=hocr_dir,
            extensions=extensions,
            auto_generate=args.generate_hocr,
            ocr_lang=args.ocr_lang,
            ocr_psm=args.ocr_psm,
            ocr_oem=args.ocr_oem,
        )
        discovered_pairs.extend(pairs_from_dir)

    if not discovered_pairs:
        raise SystemExit("No se proporcionaron pares ni imágenes para procesar.")

    seen = set()
    examples = []
    for image_path, hocr_path in discovered_pairs:
        key = (str(image_path.resolve()), str(hocr_path.resolve()))
        if key in seen:
            continue
        seen.add(key)

        example = build_layoutlm_example(
            image_path=str(image_path),
            hocr_path=str(hocr_path),
            table_bbox=bbox,
            expected_n_cols=args.expected_n_cols,
            header_regexes=args.header_regex,
            max_columns=args.max_columns,
        )
        examples.append(example.to_dict())

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False))
            fh.write("\n")

    log.info("Dataset JSONL escrito en %s (ejemplos: %d)", output_path, len(examples))


if __name__ == "__main__":
    main()
