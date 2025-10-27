from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def generate_hocr_from_image(
    image_path: str,
    output_path: Optional[str] = None,
    *,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
) -> str:
    """
    Genera un archivo HOCR a partir de una imagen usando Tesseract.

    Devuelve la ruta del archivo HOCR generado. Si `output_path` es None,
    crea un archivo junto a la imagen con la extensión `.hocr`.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow es requerido para generar HOCR.") from exc

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract es requerido para generar HOCR.") from exc

    img_path = Path(image_path)
    if output_path:
        hocr_path = Path(output_path)
    else:
        hocr_path = img_path.with_suffix(".hocr")

    log.debug("Generando HOCR para %s → %s", img_path, hocr_path)

    image = Image.open(str(img_path)).convert("RGB")
    custom_config = f"--oem {oem} --psm {psm} -c tessedit_create_hocr=1"
    hocr_bytes = pytesseract.image_to_pdf_or_hocr(
        image, extension="hocr", lang=lang, config=custom_config
    )
    hocr_path.write_bytes(hocr_bytes)
    log.info("HOCR generado: %s", hocr_path)
    return str(hocr_path)
