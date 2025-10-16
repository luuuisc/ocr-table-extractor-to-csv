# src/hocr_table_extractor/parser.py
from __future__ import annotations
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
from .structures import Token, parse_bbox, within_bbox

def _load_soup(text: str) -> BeautifulSoup:
    """
    Intenta XML (lxml-xml) y, si no hay nodos HOCR, fallback a HTML (lxml).
    """
    soup_xml = BeautifulSoup(text, "lxml-xml")
    if soup_xml.find(class_=lambda c: c and "ocr_page" in c):
        return soup_xml
    return BeautifulSoup(text, "lxml")

def parse_hocr_words(hocr_path: str,
                     table_bbox: Optional[Tuple[int,int,int,int]] = None
                     ) -> List[Token]:
    """
    Extrae tokens de palabras con bbox y página.
    NO asume anidamiento word→line; recorre todas las `ocrx_word` por página.
    """
    with open(hocr_path, "r", encoding="utf-8") as f:
        raw = f.read()
    soup = _load_soup(raw)

    tokens: List[Token] = []
    pages = soup.find_all(class_=lambda c: c and "ocr_page" in c)

    for pi, page in enumerate(pages, start=1):
        words = page.find_all(class_=lambda c: c and "ocrx_word" in c)

        # mapear line_id si el bbox de la palabra cae en el bbox de una ocr_line
        line_spans = page.find_all(class_=lambda c: c and "ocr_line" in c)
        line_boxes: List[Tuple[str, Tuple[int,int,int,int]]] = []
        for li, line in enumerate(line_spans):
            lid = line.get("id") or f"page_{pi}_line_{li+1}"
            lb = parse_bbox(line.get("title", ""))
            if lb:
                line_boxes.append((lid, lb))

        for w in words:
            bb = parse_bbox(w.get("title", ""))
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            if table_bbox and not within_bbox(table_bbox, x1, y1, x2, y2):
                continue

            text = (w.get_text() or "").strip()
            if not text:
                continue

            line_id = None
            for lid, (Lx1, Ly1, Lx2, Ly2) in line_boxes:
                if (x1 >= Lx1 and y1 >= Ly1 and x2 <= Lx2 and y2 <= Ly2):
                    line_id = lid
                    break

            tokens.append(Token(text=text, page=pi, x1=x1, y1=y1, x2=x2, y2=y2, line_id=line_id))

    return tokens
