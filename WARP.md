# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project summary
- Purpose: Extract tables from HOCR (HTML OCR) into CSV using multiple layout strategies.
- Language: Python (src/hocr_table_extractor).
- Key entrypoint: hocr_to_csv in src/hocr_table_extractor/main.py.

Setup and dependencies
- Required packages: beautifulsoup4, lxml, numpy.
- Create a virtualenv and install deps:
  ```bash path=null start=null
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install beautifulsoup4 lxml numpy
  ```

Common commands
- Run the extractor (dynamic layout):
  ```bash path=null start=null
  python -c "from hocr_table_extractor import hocr_to_csv; hocr_to_csv('input.hocr','output.csv', layout='dynamic')"
  ```
- Run with a table bounding box (x1,y1,x2,y2):
  ```bash path=null start=null
  python -c "from hocr_table_extractor import hocr_to_csv; hocr_to_csv('input.hocr','output.csv', table_bbox=(100,200,1500,2200), layout='dynamic')"
  ```
- Financial layout (3 columns: label + two numeric years):
  ```bash path=null start=null
  python -c "from hocr_table_extractor import hocr_to_csv; hocr_to_csv('fin.hocr','fin.csv', layout='financial')"
  ```
- Generic layout (fixed columns; optional header detection via regexes):
  ```bash path=null start=null
  python - <<'PY'
  from hocr_table_extractor import hocr_to_csv
  hocr_to_csv('input.hocr','generic.csv', expected_n_cols=4, header_regexes=[r'cuenta|descripcion', r'\b20\d{2}\b'], layout='generic')
  PY
  ```
- Professional layout (spatial grid + hierarchy inference):
  ```bash path=null start=null
  python -c "from hocr_table_extractor.main import hocr_to_csv; hocr_to_csv('input.hocr','pro.csv', layout='professional')"
  ```
- Logging (to see pipeline details):
  ```bash path=null start=null
  python - <<'PY'
  import logging; logging.basicConfig(level=logging.INFO)
  from hocr_table_extractor import hocr_to_csv
  hocr_to_csv('input.hocr','output.csv', layout='dynamic')
  PY
  ```

Build, lint, tests
- Build: no packaging files present (no pyproject/setup). Use the module directly from src/.
- Lint: no linter configs in repo.
- Tests: no tests/ folder or config present.

High-level architecture
The pipeline is organized as composable stages. The layout mode selects the middle stage, but the parse → line grouping → merge/export flow is consistent.

- Entry and orchestration (main.py)
  - hocr_to_csv(hocr_path, csv_path, table_bbox=None, expected_n_cols=None, header_regexes=None, layout='dynamic')
  - Drives the pipeline and switches among layouts: dynamic, financial, generic, professional.
  - Always writes CSV; in dynamic mode also writes a numeric-normalized CSV alongside (*.num.csv).

- Parsing (parser.py, structures.py)
  - parse_hocr_words reads HOCR with BeautifulSoup (XML first, then HTML fallback), extracts ocrx_word tokens with bbox and page; optionally crops by table_bbox.
  - structures.Token stores text, page, bbox, and optional line_id; helpers for bbox parsing and overlap/containment.

- Line grouping (lines.py)
  - Groups tokens into visual lines. If HOCR provides ocr_line ids, uses them; otherwise infers by vertical overlap.
  - Produces line dicts with page, inferred bbox, and ordered tokens.

- Column detection and assignment (three strategies)
  - Dynamic (column_model.py + assign_dynamic.py)
    - infer_numeric_columns_from_lines: robustly estimates numeric column x-intervals using token span clustering; optionally detects year headers to name columns.
    - assign_dynamic: merges adjacent tokens into spans, classifies numeric vs text, builds rows with one text label column + N numeric columns.
  - Financial (assign_financial.py)
    - Merges adjacent tokens into spans, filters numeric spans (supports $, commas, parentheses for negatives, and solitary '-'), assigns the two rightmost numeric spans to columns; the rest left-of-first become the label.
  - Generic (columns.py + assign.py + rows.py)
    - columns.estimate_columns builds a vertical projection profile over all tokens to estimate column intervals, with optional coercion to expected_n_cols.
    - assign.assign_words_to_columns assigns tokens to nearest interval by x-center.

- Row merging (rows.py)
  - merge_financial_rows: intelligent adjacent-line merge tuned for label wrapping and value continuation; avoids merging when both lines contain values.
  - merge_lines_into_rows: uses a horizontal projection profile to split into row intervals and merges cell content within each row.
  - detect_header_row: optional header detection (generic) via heuristics or provided regexes.

- Postprocess and export (postprocess.py, exporters.py)
  - postprocess.fill_missing_labels_and_clean: handles section lines ending with ':', normalizes '-' to '0', assigns labels like 'Total <section>' when appropriate.
  - exporters.rows_to_csv writes CSV; rows_to_csv_numeric normalizes numeric strings (handles parentheses for negatives, removes separators) and writes a parallel CSV.

- Spatial grid path (professional layout)
  - spatial.py defines BBox, SpatialWord, TableGrid.
  - grid_builder.py groups SpatialWord into lines, estimates column intervals via vertical profile, builds a preliminary grid, detects a header row, and adds hierarchy columns from indentation.
  - cleaners.process_grid_data can apply per-cell cleaning before export.

Operational notes
- HOCR input: expects classes like ocr_page, ocr_line, and ocrx_word with bbox in title attributes; parser falls back gracefully when line ids are absent.
- table_bbox: optional crop to restrict extraction to a known table rectangle (pixels).
- Layout selection: choose 'dynamic' for general financial-like tables with numeric columns; 'financial' for strict 3-column balance sheets; 'generic' for arbitrary fixed-column tables; 'professional' for spatial hierarchy extraction.
- Outputs: dynamic writes two files: output.csv and output.num.csv (numbers normalized as text). Other modes write a single CSV.
