# Arquitectura del Extractor de Tablas HOCR

## 1. Filosofía de Diseño

El núcleo de este proyecto es un diseño modular y extensible. El objetivo es poder extraer tablas de documentos HOCR, que son inherentemente ruidosos y variables, utilizando diferentes estrategias (denominadas "layouts") que pueden ser seleccionadas en tiempo de ejecución.

Cada layout representa una filosofía diferente para resolver el problema, permitiendo al usuario elegir la más adecuada para su tipo de documento o experimentar con nuevos enfoques sin alterar el código existente.

## 2. Estructura del Proyecto

El código fuente principal reside en `src/hocr_table_extractor/`. A continuación se detalla el propósito de cada módulo clave:

- **`run.py`**: Punto de entrada principal. Define la Interfaz de Línea de Comandos (CLI) usando `argparse` y orquesta la llamada al pipeline de extracción.
- **`main.py`**: El "cerebro" del extractor. Contiene la función `hocr_to_csv` que, basándose en el layout elegido, invoca la secuencia correcta de módulos para procesar el archivo.
- **`parser.py`**: Módulo de bajo nivel responsable de parsear el archivo HOCR y extraer una lista de "Tokens" (palabras) con su texto y coordenadas.
- **`structures.py` / `spatial.py`**: Definen las clases de datos fundamentales, como `Token`, `SpatialWord` y `TableGrid`, que representan los elementos del documento.
- **`dataset_builder.py` / `dataset_cli.py`**: Generan datasets JSONL con etiquetas `HEADER_COL_i` / `BODY_COL_i` usando el layout `generic` como maestro. Se emplean para fine-tunear LayoutLM.
- **`train_layoutlm.py`**: Script de entrenamiento que monta un `Trainer` de Hugging Face para LayoutLMv3, consumiendo los JSONL anteriores y produciendo checkpoints reutilizables.

---

## 3. Flujo de Datos y Estrategias de Layout

### 3.1. Layout `generic` (El Punto de Referencia)

Este layout es, en la versión actual, el más robusto y el que produce resultados más consistentes. Su éxito se basa en el uso de técnicas clásicas de análisis de documentos.

**Pipeline:**
1.  **`parse_hocr_words`**: Se extraen todos los tokens del HOCR.
2.  **`estimate_columns` (`columns.py`)**: Utiliza un **perfil de proyección vertical**. Imagina "aplastar" todo el texto de la página en un solo eje horizontal. Los espacios en blanco que persisten a lo largo de la página se convierten en "valles" en el perfil, y los centros de estos valles son los separadores de columnas. Es una técnica muy robusta.
3.  **`assign_words_to_columns` (`assign.py`)**: Asigna cada palabra a la columna que le corresponde según los límites definidos en el paso anterior.
4.  **`merge_lines_into_rows` (`rows.py`)**: De forma similar al paso 2, utiliza un **perfil de proyección horizontal** para encontrar los espacios en blanco entre las filas. Esto le permite fusionar inteligentemente texto que ocupa varias líneas (como descripciones largas) en una única fila lógica de la tabla.

### 3.2. Layouts `financial` y `dynamic`

Estos layouts son más heurísticos y fueron los primeros enfoques implementados.
- **`financial`**: Asume una estructura rígida de 3 columnas (Cuenta, Valor 1, Valor 2). Es rápido pero poco flexible.
- **`dynamic`**: Intenta inferir qué columnas son numéricas y cuáles son de texto para separarlas. Es un paso intermedio hacia un análisis más avanzado.

### 3.3. Layout `professional` (Híbrido y Experimental)

Este layout es el resultado de nuestro proceso iterativo y su objetivo es ser el más avanzado, combinando las mejores técnicas de los otros layouts con nuevas funcionalidades.

**Pipeline Híbrido:**
1.  **Detección de Columnas Robusta**: Adopta el excelente método de **perfil de proyección vertical** del layout `generic`.
2.  **Construcción de Filas Robusta**: Adopta el método de **perfil de proyección horizontal** del layout `generic` para fusionar correctamente las filas.
3.  **Enriquecimiento Avanzado**: Sobre la tabla ya bien estructurada, aplica sus propias funcionalidades:
    - **Reconocimiento de Jerarquía**: Analiza la indentación de la primera columna para añadir nuevas columnas (`Level_1`, `Level_2`, etc.) que representan la estructura anidada del documento.
    - **Limpieza de Datos**: Convierte automáticamente texto numérico de formato contable (ej. `(1,234.50)`) a números estándar.
    - **Detección de Cabecera Inteligente**: Busca la fila que más probablemente sea la cabecera en lugar de asumir que es la primera.

**Estado Actual y Puntos a Mejorar:**
La fusión de las dos lógicas (la construcción de `generic` y el enriquecimiento de `professional`) es compleja. La interacción entre la fusión de filas de `generic` y el algoritmo de detección de jerarquía de `professional` puede no ser perfecta aún, lo que podría explicar por qué en ciertos documentos el resultado de `generic` sigue siendo visualmente más limpio. La base para el layout más potente está sentada, pero requiere más ajustes finos.

### 3.4. Layout `transformers` (LayoutLMv3 + OCR)

Nuevo pipeline que opera directamente sobre la **imagen**:
1. `pytesseract` obtiene palabras y bounding boxes (filtradas por confianza y, opcionalmente, por `table_bbox`).
2. `LayoutLMv3Processor` + `LayoutLMv3ForTokenClassification` (por defecto `microsoft/layoutlmv3-base`, pero puede cargarse un checkpoint fine-tuned) predicen etiquetas token a token (`HEADER_COL_i`, `BODY_COL_i`, `OTHER`).
3. Las etiquetas se consolidan en palabras, se proyectan perfiles verticales/horizontales y se reconstruyen filas y columnas. Si el modelo no produce señal suficiente, se aplica un fallback heurístico (`grid_builder` + `process_grid_data`).

**Entrenamiento**: el dataset se autogenera a partir de HOCR + CSV de referencia (`dataset_cli`). `train_layoutlm.py` expone un flujo reproducible que guarda modelo y processor para reutilizarlos con el layout `transformers`.
