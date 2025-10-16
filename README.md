# HOCR Table Extractor

Una herramienta modular para reconstruir datos tabulares a partir de archivos HOCR y exportarlos a formato CSV. El proyecto utiliza un enfoque personalizable para analizar la estructura de la tabla, permitiendo al usuario elegir entre diferentes estrategias de extracción.

## Características

- **Múltiples Modos de Extracción:**
  - `dynamic`: Un modo avanzado que infiere automáticamente las columnas numéricas y de texto. Ideal para informes financieros y tablas complejas.
  - `financial`: Una heurística especializada para balances con el formato `Cuenta | Valor Año 1 | Valor Año 2`.
  - `generic`: Un modo versátil que detecta columnas basándose en los espacios verticales, utilizando perfiles de proyección. Útil para tablas genéricas bien estructuradas.
- **Interfaz de Línea de Comandos (CLI):** `run.py` proporciona una CLI completa para controlar el proceso de extracción.
- **Personalización:** Permite especificar el área de la tabla (`--bbox`), el número de columnas esperado (`--expected-n_cols`), y otros parámetros para afinar la extracción.
- **Manejo de Dependencias:** Incluye un archivo `requirements.txt` para una fácil instalación.

## Guía de Inicio Rápido

1. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el extractor:**

   El script principal es `run.py`. Debes proporcionar la ruta al archivo HOCR de entrada y la ruta para el archivo CSV de salida.

   ```bash
   python run.py /ruta/al/input.hocr /ruta/al/output.csv
   ```

### Ejemplos de Uso

- **Usar el modo genérico:**

  ```bash
  python run.py input.hocr output.csv --layout generic
  ```

- **Especificar el número de columnas en modo genérico:**

  ```bash
  python run.py input.hocr output.csv --layout generic --expected-n-cols 4
  ```

- **Limitar la extracción a un área específica (Bounding Box):**

  Si la página contiene mucho "ruido" fuera de la tabla, puedes especificar las coordenadas `x1 y1 x2 y2` del área que te interesa.

  ```bash
  python run.py input.hocr output.csv --bbox 80 220 1750 2450
  ```

- **Ajustar el nivel de logs para depuración:**

  ```bash
  python run.py input.hocr output.csv --loglevel DEBUG
  ```

Para ver todas las opciones disponibles, ejecuta:

```bash
python run.py --help
```

## Estructura del Proyecto

- `run.py`: Punto de entrada principal y CLI.
- `requirements.txt`: Lista de dependencias del proyecto.
- `src/hocr_table_extractor/`:
  - `main.py`: Orquesta el flujo de extracción.
  - `parser.py`: Parsea el HOCR y extrae tokens.
  - `lines.py`: Agrupa tokens en líneas.
  - `columns.py`: Estima las columnas (usando perfiles de proyección).
  - `column_model.py`: Infiere columnas numéricas para el modo `dynamic`.
  - `assign.py`, `assign_dynamic.py`, `assign_financial.py`: Asignan palabras a celdas.
  - `rows.py`: Fusiona líneas en filas.
  - `postprocess.py`: Realiza limpieza en los datos extraídos.
  - `exporters.py`: Exporta los datos a CSV.
  - `structures.py`: Clases de datos y utilidades de Bounding Box.

## Dependencias

- Python 3.9+
- `beautifulsoup4`
- `lxml`
- `numpy`
