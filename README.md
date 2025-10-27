# HOCR Table Extractor

<p align="center">
  <img src="images/app.png" alt="App UI" width="720">
</p>

Una herramienta modular para reconstruir datos tabulares a partir de archivos HOCR y exportarlos a formato CSV. El proyecto utiliza un enfoque personalizable para analizar la estructura de la tabla, permitiendo al usuario elegir entre diferentes estrategias de extracción.

> Para conocer en detalle el flujo LayoutLMv3 (dataset → entrenamiento → inferencia), consulta **[LAYOUTLM_WORKFLOW.md](LAYOUTLM_WORKFLOW.md)**.

## Estado del Proyecto y Guía para Evaluación

Este proyecto se ha desarrollado de forma iterativa, explorando diferentes algoritmos para la extracción de tablas. Como resultado, existen múltiples "layouts" o estrategias de extracción.

**Para fines de evaluación, se recomienda probar el layout `generic`**, ya que actualmente es el que produce los resultados más robustos y consistentes en una amplia variedad de documentos. Los otros layouts (`dynamic`, `financial`, `professional`) representan diferentes etapas de la evolución del proyecto y, aunque funcionales, pueden no ser tan estables.

Para una explicación técnica detallada de cada layout y la arquitectura general, por favor consulte el archivo `ARCHITECTURE.md`.

## Guía de Inicio Rápido

1.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Ejecutar el extractor (Modo Recomendado):**

    Para obtener el mejor resultado, utilice el layout `generic`, de preferencia especificando el número de columnas que se esperan

    ```bash
    python run.py ruta/al/output.csv --hocr_path ruta/al/input.hocr --layout generic --expected-n-cols 4
    ```
> [!IMPORTANT]
> Los archivos `.hocr` deben estar en la raíz del proyecto

## Opciones Avanzadas y Layouts

### Layout `generic` (Recomendado)

Es el layout más fiable. Utiliza un doble análisis de perfiles de proyección (vertical para columnas y horizontal para filas) para reconstruir la tabla. Es la implementación de referencia.

-   **Uso:**
    ```bash
    python run.py output.csv --hocr_path input.hocr --layout generic
    ```
-   **Especificar número de columnas (opcional):**
    ```bash
    python run.py output.csv --hocr_path input.hocr --layout generic --expected-n-cols 4
    ```
-   **Ejemplo incluido:** `salida_genericC.csv` muestra el resultado obtenido sobre `1C.hocr` con `--expected-n-cols 4`.

### Layout `professional` (Experimental)

Este layout es un híbrido que intenta combinar la robusta construcción de tablas del `generic` con características avanzadas como la detección de jerarquía y la limpieza de datos. Aunque es funcional, su integración aún está en desarrollo y puede producir resultados inesperados.

-   **Uso:**
    ```bash
    python run.py output.csv --hocr_path input.hocr --layout professional
    ```

### Layout `transformers` (LayoutLMv3-base)

Pipeline basado en `microsoft/layoutlmv3-base` + OCR de Tesseract. Reconstruye la tabla directamente desde la imagen, etiquetando cada palabra con el modelo y agrupándola en filas/columnas. Cuando el modelo no entrega señal suficiente, el flujo cae en un fallback heurístico tipo grid (resultados similares al layout `generic`).

-   **Dependencias adicionales:** `torch`, `transformers`, `pillow`, `pytesseract` y el ejecutable de Tesseract.
-   **Uso:**
    ```bash
    python run.py balance.csv --image input.jpeg --layout transformers \
      --transformer-model fine_tuned_layoutlm_checkpoint \
      --transformer-max-cols 4
    ```
-   **Parámetros útiles:**
    - `--transformer-model`: checkpoint Hugging Face o ruta local (por defecto `microsoft/layoutlmv3-base`).
    - `--transformer-ocr-lang`: idioma para Tesseract (default `eng`).
    - `--transformer-max-cols`: límite superior de columnas reconstruidas (fallback a 6).
    - `--bbox x1 y1 x2 y2`: recorta la región de la tabla antes de correr OCR/modelo.
-   **Salida:** CSV con cabecera detectada y filas del balance; opcionalmente genera fallback inteligente si el modelo no cubre la tabla completa.
-   **Estado actual:** con el dataset de ejemplo, el layout `generic` sigue entregando la tabla más limpia (`salida_genericC.csv`). El modelo LayoutLM mejora conforme se entrene con más ejemplos (ver nota abajo).

### Otros Parámetros

-   **Limitar a un área específica (Bounding Box):**

    ```bash
    python run.py output.csv --hocr_path input.hocr --bbox 80 220 1750 2450
    ```

-   **Ajustar el nivel de logs para depuración:**

    ```bash
    python run.py output.csv --hocr_path input.hocr --loglevel DEBUG
    ```

### Generar dataset para LayoutLM

Se añadió una utilidad para crear ejemplos etiquetados (`JSONL`) usando el layout `generic` como maestro y así fine-tunear `LayoutLMv3`. Este paso es opcional si solo se quiere usar el checkpoint base, pero recomendable para lograr buenas predicciones.

```bash
python -m hocr_table_extractor.dataset_cli \
  --pair 1C.jpeg:1C.hocr \
  --pair 1E.jpeg:1E.hocr \
  --output data/layoutlm_dataset.jsonl \
  --expected-n-cols 4 \
  --header-regex 'cuenta|descripcion' \
  --header-regex '20\d{2}'
```

-   **Procesar carpetas completas (auto-generar HOCR):**

    ```bash
    python -m hocr_table_extractor.dataset_cli \
      --image-dir train_images \
      --generate-hocr \
      --output data/layoutlm_dataset_auto.jsonl \
      --expected-n-cols 4
    ```

Cada línea del JSONL contiene:
- `words`, `bboxes` (normalizados a 0-1000) y `labels` (`HEADER_COL_i`, `BODY_COL_i`, `OTHER`)
- Información estructural (`row_ids`, `col_ids`, `table_header`, `table_rows`) para auditoría

### Entrenar LayoutLM

El script `train_layoutlm.py` orquesta el fine-tuning usando los JSONL anteriores.

```bash
python -m hocr_table_extractor.train_layoutlm \
  --train-jsonl data/layoutlm_dataset.jsonl \
  --output-dir checkpoints/layoutlm-finetuned \
  --batch-size 2 \
  --num-epochs 5 \
  --learning-rate 3e-5 \
  --fp16
```

- Usa `--eval-jsonl` para pasar un set de validación; si se omite, divide automáticamente el train.
- El checkpoint resultante puede usarse con `--transformer-model`.
- Si el script pide `accelerate`, instala `pip install 'accelerate>=0.26.0'`.
- Guía paso a paso (entorno, dataset, entrenamiento, inferencia, troubleshooting): **[LAYOUTLM_WORKFLOW.md](LAYOUTLM_WORKFLOW.md)**

Para ver todas las opciones disponibles, ejecuta:

```bash
python run.py --help
```

## Dependencias

-   Python 3.9+
-   `beautifulsoup4`
-   `lxml`
-   `numpy`
-   *(Layout `transformers`)* `torch`, `transformers`, `pillow`, `pytesseract` + binario de Tesseract

```bash
python run.py salida_genericT_P.csv --hocr_path 3T.hocr --layout generic --expected-n-cols 4
```
