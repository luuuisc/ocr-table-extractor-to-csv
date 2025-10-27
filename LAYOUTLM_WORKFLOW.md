# LayoutLMv3 Workflow Guide

Este documento describe el flujo completo para generar datasets, entrenar y ejecutar el nuevo layout `transformers`, basado en LayoutLMv3, a partir de cero en cualquier equipo (macOS/Windows/Linux).

## 1. Preparación del entorno

1. **Crear y activar un entorno virtual (Python ≥ 3.10 recomendado)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate           # macOS / Linux
   .venv\Scripts\activate              # Windows PowerShell
   ```

2. **Instalar dependencias del proyecto**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > El layout `transformers` necesita `torch`, `transformers`, `pillow`, `pytesseract`, `accelerate` y el ejecutable de Tesseract instalado en el sistema.

3. **Verificar soporte del backend (CPU/GPU)**

   ```bash
   python - <<'PY'
   import torch
   print("MPS disponible:", torch.backends.mps.is_available())
   print("CUDA disponible:", torch.cuda.is_available())
   PY
   ```

   - macOS Apple Silicon: se usará `torch.backends.mps` (activar `PYTORCH_ENABLE_MPS_FALLBACK=1`).
   - NVIDIA GPU: asegurarse de tener CUDA Toolkit y drivers correctos.

4. **Añadir el paquete al `PYTHONPATH` o instalarlo en editable**

   ```bash
   export PYTHONPATH=src                     # macOS / Linux
   setx PYTHONPATH "src"                     # Windows (PowerShell)
   # o
   pip install -e src
   ```

## 2. Generación del dataset supervisado

El CLI `dataset_cli` toma pares `IMAGEN:HOCR`, aplica el layout `generic` como profesor y produce un JSONL con los tokens, cajas y etiquetas (`HEADER_COL_i`, `BODY_COL_i`, `OTHER`), listo para fine-tuning.

### 2.1. Usando imágenes con HOCR existente

```bash
PYTHONPATH=src python -m hocr_table_extractor.dataset_cli \
  --pair 1C.jpeg:1C.hocr \
  --pair 1E.jpeg:1E.hocr \
  --pair 3T.jpeg:3T.hocr \
  --output data/layoutlm_train.jsonl \
  --header-regex 'cuenta|descripcion' \
  --header-regex '20\d{2}'
```

Opciones útiles:
- `--bbox x1 y1 x2 y2`: recorta la región de la tabla antes de etiquetar.
- `--expected-n-cols`: fuerza el número de columnas del maestro (opcional).

Verificar el resultado:

```bash
head -n 1 data/layoutlm_train.jsonl | jq
```

### 2.2. Usando un directorio de imágenes (auto-generación de HOCR)

Si cuentas con un lote de imágenes sin HOCR (p. ej. `train_images/`), el CLI puede generarlos automáticamente con Tesseract:

```bash
PYTHONPATH=src python -m hocr_table_extractor.dataset_cli \
  --image-dir train_images \
  --generate-hocr \
  --ocr-lang eng \
  --output data/layoutlm_auto.jsonl \
  --expected-n-cols 4 \
  --header-regex 'cuenta|descripcion' \
  --header-regex '20\d{2}'
```

Opciones adicionales:
- `--hocr-dir DIR`: busca/guarda los `.hocr` en otra carpeta.
- `--extensions .jpeg .jpg .png`: restringe las extensiones a procesar.
- `--ocr-psm / --ocr-oem`: ajusta configuraciones de Tesseract.

Puedes combinar ambos enfoques:

```bash
PYTHONPATH=src python -m hocr_table_extractor.dataset_cli \
  --pair 1C.jpeg:1C.hocr \
  --pair 1E.jpeg:1E.hocr \
  --pair 3T.jpeg:3T.hocr \
  --image-dir train_images \
  --generate-hocr \
  --output data/layoutlm_full.jsonl
```

## 3. Entrenamiento del modelo (fine-tuning)

En macOS Apple Silicon:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHONPATH=src python -m hocr_table_extractor.train_layoutlm \
  --train-jsonl data/layoutlm_train.jsonl \
  --output-dir checkpoints/layoutlm-fin \
  --batch-size 2 \
  --num-epochs 5 \
  --learning-rate 3e-5 \
  --logging-steps 10 \
  --num-workers 0 \
  --metrics-json reports/train_log.json \
  --metrics-csv reports/train_log.csv
```

Notas:
- Para CPU puro, elimina la variable `PYTORCH_ENABLE_MPS_FALLBACK`.
- Si cuentas con GPU NVIDIA, puedes agregar `--fp16` y ajustar `--batch-size`.
- El script divide automáticamente el dataset si no se provee `--eval-jsonl`.

Al finalizar, el directorio de salida debe contener `pytorch_model.bin` (o `model.safetensors`), `config.json`, `preprocessor_config.json`, etc.
Si añades `--metrics-json/--metrics-csv`, también obtendrás el historial de pérdida/accuracy registrado por el Trainer en formatos fáciles de compartir.

## 4. Ejecución del layout `transformers`

Con el checkpoint fine-tuneado:

```bash
PYTHONPATH=src python run.py salida_transformers_1C.csv \
  --image 1C.jpeg \
  --layout transformers \
  --transformer-model checkpoints/layoutlm-fin \
  --transformer-max-cols 4 \
  --loglevel INFO
```

Parámetros relevantes:
- `--transformer-model`: ruta al checkpoint entrenado o ID de Hugging Face Hub.
- `--transformer-ocr-lang`: idioma para Tesseract (default `eng`).
- `--transformer-max-cols`: máximo de columnas reconstruidas; si se omite, se deriva automáticamente.
- `--bbox`: limita la región analizada en la imagen.

Si el modelo no produce señal suficiente, el layout recurre a un fallback heurístico basado en la rejilla espacial (`grid_builder`).

## 5. Comparación con layouts clásicos

Generar referencia con el layout `generic`:

```bash
PYTHONPATH=src python run.py salida_generic_1C.csv \
  --hocr_path 1C.hocr \
  --layout generic \
  --expected-n-cols 4
```

Comparar resultados:

```bash
diff -u salida_generic_1C.csv salida_transformers_1C.csv
```

Por ahora, el layout `generic` sigue siendo el más estable. Usa `transformers` para experimentar con el modelo fine-tuneado y detectar oportunidades de mejora.

## 6. Métricas cuantitativas (Text accuracy, MSE, R²)

Además del diff visual, puedes cuantificar la calidad mediante el script `eval_cli`:

```bash
python -m hocr_table_extractor.eval_cli \
  --reference salida_generic_1C.csv \
  --predicted salida_transformers_1C.csv \
  --report reports/eval_1C.csv \
  --json reports/eval_1C.json
```

Valores reportados:

- `text_accuracy`: proporción de celdas exactamente iguales (útil para celdas textuales).
- `MSE`, `RMSE`, `R²`: comparan columnas numéricas; se infieren automáticamente, pero puedes especificar `--numeric-columns`.

Estas métricas ayudan a argumentar mejoras frente al baseline durante una entrevista técnica.

## 6. Troubleshooting

- **`ModuleNotFoundError: hocr_table_extractor`**  
  Asegúrate de exportar `PYTHONPATH=src` o instalar el paquete con `pip install -e src`.

- **`ImportError: accelerate>=0.26.0`**  
  Instala Accelerate: `pip install 'accelerate>=0.26.0'`.

- **`UserWarning: pin_memory` en MPS**  
  Es esperado; la memoria anclada no está soportada en MPS, pero el entrenamiento continúa.

- **Predicciones en una sola columna**  
  Incrementa el dataset (más ejemplos etiquetados), revisa `--transformer-max-cols` y confirma que las etiquetas del JSONL incluyan columnas múltiples (`BODY_COL_0`, `BODY_COL_1`, etc.).

## 7. Próximos pasos sugeridos

- Ampliar el dataset con más balances y variaciones de formato.
- Evaluar la integración con detectores de celdas (p. ej. Table Transformer) para mejorar la segmentación espacial.
- Diseñar etiquetas adicionales para otros tipos de documentos (actas, licencias) y entrenar modelos multi-task.
