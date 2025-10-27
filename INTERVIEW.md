# hOCR Table Extractor

## 1. Visión General del Proyecto

Este proyecto es un sistema automatizado para la extracción de datos tabulares a partir de documentos digitalizados (imágenes o PDFs).
**Problema:** La extracción manual de datos de tablas en facturas, reportes o formularios es un proceso ineficiente, costoso y propenso a errores humanos.
**Solución:** Este pipeline toma como entrada el resultado de un motor de OCR (en formato hOCR, que incluye texto y coordenadas) y, mediante una serie de algoritmos de análisis espacial y reconstrucción geométrica, infiere la estructura de la tabla, extrae su contenido y
lo devuelve en un formato estructurado (JSON, CSV).
**Impacto:** Aumenta drásticamente la eficiencia operativa, reduce los errores y permite la integración automática de datos que antes estaban "atrapados" en documentos.

## 2. Arquitectura y Pipeline de Procesamiento

El sistema está diseñado como un pipeline modular y secuencial. Esta arquitectura desacoplada permite que cada componente sea mantenido, mejorado y probado de forma independiente.

1. **Entrada:** Un archivo `hOCR` generado por un motor de OCR como Tesseract.
2. **Parseo (`parser.py`):** Se parsea el hOCR para extraer cada palabra y sus coordenadas (`bounding box`).
3. **Reconstrucción Geométrica (`grid_builder.py`, `lines.py`, `spatial.py`):** Se agrupan las palabras en líneas y se analizan sus alineaciones verticales y horizontales para inferir las fronteras de las filas y columnas, construyendo una grilla virtual.
4. **Asignación de Contenido (`assign_.py`):** El texto de las palabras se asigna a las celdas correspondientes de la grilla. Se utiliza un **patrón de diseño de Estrategia** para manejar diferentes tipos de tablas (financieras, dinámicas, etc.).
5.  **Post-procesamiento (`cleaners.py`, `postprocess.py`):** Se limpian los datos (ej: eliminar símbolos de moneda, convertir texto a números) y se realizan ajustes estructurales (ej: fusionar celdas).
6.  **Exportación (`exporters.py`):** La tabla, ya como un objeto estructurado en memoria, se serializa a un formato de salida útil como JSON o CSV.

## 3. Desglose de Módulos (`src/hocr*table_extractor/`)

Cada archivo `.py`tiene una responsabilidad única y clara dentro del pipeline:

* `run.py`: (En la raíz) El punto de entrada principal que orquesta la ejecución del pipeline desde la línea de comandos. Gestiona los argumentos de entrada/salida y la selección de la estrategia.
* `main.py`: El orquestador principal del pipeline. Llama a cada módulo en la secuencia correcta: parseo, construcción, asignación, etc.
* `parser.py`: **Responsabilidad:** Parsear el input hOCR. Su única función es leer el formato hOCR (HTML) y transformarlo en una lista de objetos `Word`con su texto y coordenadas.
* `structures.py`: **Responsabilidad:** Definir las estructuras de datos clave. Probablemente contiene las `dataclasses`o clases`Word`, `Cell`, `Row`, y `Table`que se usan a través de toda la aplicación.
* `spatial.py`: **Responsabilidad:** Contener la lógica matemática. Es una librería de utilidades para todos los cálculos geométricos (distancias, intersecciones, alineaciones, etc.). Es un módulo puro y altamente reutilizable.
* `lines.py`: **Responsabilidad:** Agrupar palabras en líneas. Toma la lista de palabras del parser y, basándose en su superposición vertical, las agrupa en objetos `Line`.
* `rows.py`/`columns.py`: **Responsabilidad:** Identificar los límites de las filas y columnas. Usan técnicas como histogramas de proyección sobre los ejes X e Y para encontrar los espacios en blanco que definen la grilla.
* `grid_builder.py`: **Responsabilidad:** Orquestar la construcción de la grilla. Utiliza los módulos `rows`y`columns`para construir la estructura vacía de la tabla.
* `assign.py`: **Responsabilidad:** Implementar la estrategia de asignación de contenido por defecto. Define la lógica base para decidir a qué celda de la grilla pertenece cada palabra.
* `assign_financial.py`/`assign_dynamic.py`: **Responsabilidad:** Implementar estrategias de asignación especializadas. Contienen reglas específicas para tablas financieras (manejo de símbolos, paréntesis) o tablas dinámicas (con celdas fusionadas, etc.).
* `cleaners.py`: **Responsabilidad:** Limpieza a nivel de datos. Contiene funciones para limpiar el texto extraído (ej: `clean_currency()`, `text_to_number()`).
* `postprocess.py`: **Responsabilidad:** Limpieza a nivel de estructura. Realiza ajustes a la tabla ya construida, como fusionar celdas o corregir errores estructurales.
* `exporters.py`: **Responsabilidad:** Serializar el output. Convierte el objeto final `Table`a formatos de archivo como JSON o CSV.

## 4. Decisiones Clave de Diseño (Puntos para la Entrevista)

* **Enfoque Algorítmico vs. ML:** Se eligió un enfoque geométrico por su **determinismo, explicabilidad, alto rendimiento y por no requerir datos de entrenamiento**. Es una base robusta y eficiente.
* **Desacoplamiento del OCR:** El sistema consume hOCR, no imágenes. Esto lo hace agnóstico al motor de OCR, permitiendo cambiarlo en el futuro sin alterar la lógica de extracción.
* **Patrón de Diseño "Estrategia":** El uso de múltiples archivos`assign*_.py` permite extender fácilmente el sistema para nuevos tipos de documentos sin modificar el código existente, demostrando adhesión a los principios SOLID (Open/Closed Principle).

## 5. Hoja de Ruta y Futuras Mejoras (Visión de Producto)

Este proyecto es una base sólida, pero está diseñado para evolucionar.


**Fase 1: Industrialización:**
**Testing:** Implementar una suite de tests completa con `pytest` (unitaria, integración, regresión).
**Configuración Centralizada:** Mover umbrales y parámetros a un archivo `config.yaml`.
**Logging y Monitoreo:** Añadir logging estructurado para trazabilidad y monitoreo de rendimiento.

**Fase 2: Inteligencia y Escalabilidad:**
**API y Contenerización:** Exponer el pipeline como un microservicio API con **FastAPI** y empaquetarlo con **Docker** para un despliegue sencillo en la nube (AWS/GCP).
**Clasificador de Estrategias:** Desarrollar un modelo simple que clasifique el documento de entrada y seleccione automáticamente la estrategia de asignación (`financial`, `dynamic`, etc.).
**Validación con LLMs:** Integrar un LLM para que actúe como una capa de **verificación y corrección** sobre el JSON extraído, mejorando la precisión con un coste computacional controlado.

**Fase 3: Expansión de Capacidades:**
**Manejo de Casos Complejos:** Desarrollar módulos para la corrección de rotación de documentos y el manejo de tablas con celdas fusionadas.
**Benchmarking:** Realizar un análisis comparativo entre el enfoque actual, un enfoque híbrido (algoritmo + LLM) y un enfoque puro de LLM multimodal para determinar la mejor solución para cada tipo de documento.

## 6. Preguntas y Respuestas Estratégicas (Nivel Senior)

 Esta sección simula preguntas clave que un entrevistador haría para evaluar la profundidad estratégica y la visión de un candidato.

  <br>
  P1: ¿Por qué elegiste una arquitectura algorítmica/geométrica en lugar de un modelo de Deep Learning de extremo a extremo?

   * Respuesta: "Fue una decisión de diseño deliberada. Un enfoque algorítmico nos da determinismo (esencial para la depuración), alto rendimiento con baja latencia y bajo coste computacional. Además, evita la necesidad de un costoso dataset de entrenamiento anotado. Este enfoque
     establece una base sólida y predecible, que luego puede ser aumentada con ML/LLMs para manejar los casos más complejos, combinando lo mejor de ambos mundos."

  <br>

  P2: ¿Cuál es el talón de Aquiles de este sistema? ¿Qué harías si un cliente te trae un documento que lo hace fallar constantemente?

   * Respuesta: "Su principal debilidad son las tablas muy irregulares o las que no tienen una estructura de grilla clara (ej: celdas fusionadas de forma compleja). Si un documento falla, mi proceso sería: 1) Análisis de Causa Raíz: Usar el logging para identificar qué parte del pipeline
     falla (¿la agrupación de líneas? ¿la detección de columnas?). 2) Añadir a Regresión: Incorporar ese documento a una suite de tests de regresión para que, una vez arreglado, no vuelva a fallar. 3) Evaluar la Solución: Decidir si el fallo se puede corregir con un ajuste en los
     algoritmos existentes o si justifica el desarrollo de una nueva estrategia de asignación específica para ese tipo de documento. Esto transforma un fallo en una mejora del sistema."

  <br>

  P3: Más allá de la precisión técnica, ¿cómo medirías el éxito de este proyecto en términos de negocio (KPIs)?

   * Respuesta: "El éxito se mide por el valor que aporta. Mis KPIs clave serían: 1) Reducción del Tiempo de Procesamiento por Documento: Comparar el tiempo manual vs. el automático. 2) Tasa de Extracción Directa (Straight-Through Processing Rate): El porcentaje de documentos que se
     procesan correctamente sin necesidad de intervención humana. El objetivo es maximizar este número. 3) Reducción de la Tasa de Errores: Medir la disminución de errores de entrada de datos en comparación con el proceso manual. 4) Coste por Documento Procesado: Calcular el coste
     computacional y demostrar que es significativamente menor que el coste del trabajo manual."

  <br>

  P4: ¿Cómo escalarías esta solución para procesar 1 millón de documentos al día?

   * Respuesta: "La arquitectura está diseñada para ser stateless, lo que la hace ideal para el escalado horizontal. Mi plan sería: 1) Contenerizar la aplicación con Docker. 2) Usar una Cola de Mensajes (como AWS SQS o RabbitMQ) para gestionar la carga de trabajo. 3) Desplegar un clúster 
     de workers autoescalable (usando Kubernetes o AWS ECS/Fargate) que consuman trabajos de la cola en paralelo. Si un worker falla, la cola asegura que el trabajo no se pierda. Este patrón de productor/consumidor es robusto, resiliente y puede escalar a prácticamente cualquier volumen."


  <br>

## 7. Glosario de Conceptos Clave de IA

Esta sección demuestra un entendimiento del ecosistema de IA actual, conectando el proyecto con las tendencias de la industria.

* IA Generativa (GenAI):
       * Se refiere a modelos de IA que pueden crear contenido nuevo y original en lugar de solo analizar o clasificar datos existentes. El contenido puede ser texto, imágenes, código, música, etc. Este proyecto, en su estado actual, es de IA "extractiva", pero la hoja de ruta lo
         evoluciona al integrarlo con GenAI.

* Modelos de Lenguaje Grandes (LLMs):
       * Son el tipo más común de IA Generativa, entrenados en cantidades masivas de texto. Son la base de sistemas como ChatGPT. Su poder reside en su profunda comprensión del lenguaje, lo que les permite resumir, traducir, generar texto y razonar sobre el contenido. En nuestro
         proyecto, proponemos usarlos para corregir y validar la data extraída.

* IA Agéntica / Agentes de IA:
       * Un agente de IA es un sistema que va más allá de un modelo pasivo. Es un sistema autónomo que opera en un ciclo para alcanzar un objetivo. Sus componentes son:
           * Objetivo: Una meta definida (ej: "Extraer los datos de la tabla de este documento").
           * Herramientas: Un conjunto de capacidades que puede usar (ej: read_file, run_ocr_tool, run_table_extractor_pipeline).
           * Planificación/Razonamiento: Un "cerebro" (generalmente un LLM) que, dado el objetivo, elige la mejor herramienta a usar en cada momento.
           * Ciclo de Acción: Opera en un bucle de Observar -> Pensar -> Actuar hasta que el objetivo se cumple. Nuestro proyecto podría convertirse en una herramienta clave para un agente de IA más grande encargado de procesar documentos.

* RAG (Retrieval-Augmented Generation):
       * Es una técnica para hacer que los LLMs sean más fiables y precisos. En lugar de solo responder desde su conocimiento pre-entrenado, el sistema primero recupera información relevante de una base de conocimientos externa (como los datos extraídos de nuestros documentos) y luego le
         pasa esa información al LLM junto con la pregunta del usuario. Esto "ancla" la respuesta del LLM en hechos concretos, reduciendo alucinaciones. Nuestro extractor es la herramienta perfecta para poblar la base de conocimientos de un sistema RAG.

* Multimodality:
       * La capacidad de un modelo de IA para procesar y entender información de múltiples tipos de datos (modalidades) simultáneamente, como texto, imágenes y audio. Un modelo multimodal podría, en teoría, tomar la imagen de nuestro documento directamente y producir el JSON de la tabla,
         combinando la visión por computadora y el razonamiento en un solo paso. La hoja de ruta del proyecto incluye el benchmarking de estos modelos contra nuestro enfoque especializado.
