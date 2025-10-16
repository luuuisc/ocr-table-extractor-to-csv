# HOCR TABLE EXTRACTOR

Modular extractor to reconstruct tabular data from an HOCR file and export it to CSV.

## Features
- Robust parsing of HOCR using BeautifulSoup (lxml parser).
- Line grouping (using `<span class="ocr_line">` when available, or inferred).
- Column detection via robust gap histogram (percentile threshold).
- Row merging for multi-line cells.
- Heuristics configurable via CLI flags or Python API.

## Quickstart
```bash
python -m hocr_table_extractor.main /path/to/input.hocr /path/to/output.csv
```

Common flags:
```bash
python -m hocr_table_extractor.main input.hocr output.csv   --ncols 6   --header "(?i)cuenta" "(?i)saldo" "(?i)total"   --colq 94   --rowmerge 1.3
```

Limit detection to a known table area (bbox from HOCR coords):
```bash
python -m hocr_table_extractor.main input.hocr output.csv --bbox 80 220 1750 2450
```

## Project layout
- `src/hocr_table_extractor/`
  - `structures.py` : dataclasses y utilidades de bbox/overlaps
  - `parser.py`     : parser de HOCR a tokens
  - `lines.py`      : agrupación de tokens en líneas
  - `columns.py`    : estimación de columnas (gutters por gaps)
  - `assign.py`     : asignación de tokens a columnas
  - `rows.py`       : unión de líneas en filas, detección de encabezados
  - `exporters.py`  : exportación a CSV
  - `main.py`       : función orquestadora y CLI

## Requirements
- Python 3.10+
- `beautifulsoup4`
- `lxml`
- `numpy`
- `pandas`

Install dev deps:
```bash
pip install -r requirements.txt
```

## Notes
- Adjust `--colq` (percentile for large gaps) and `--rowmerge` per document.
- If table region is known, provide `--bbox` to reduce noise.
- For highly regular statements, pass `--ncols` to stabilize column widths.
