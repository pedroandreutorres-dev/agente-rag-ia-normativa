# Consultor RAG: Protección de Datos (RGPD / EDPB) e Inteligencia Artificial Agéntica

**Repositorio Oficial — Proyecto Final de Máster en Inteligencia Artificial y Ciencia de Datos**  
**Dominio de Especialización:** Derecho Tecnológico, Cumplimiento Normativo (RGPD), Dictámenes del EDPB y Gobernanza de Sistemas de Inteligencia Artificial Agéntica  
**Pila Tecnológica:** Python 3.10+, LangChain, LangGraph V4 (StateGraph + Router condicional), Google Gemini API (`gemini-3.1-flash-lite`), ChromaDB (MMR), MemorySaver y Streamlit  
**🚀 Aplicación Web en Vivo (Streamlit Cloud):** [https://agente-rag-ia-normativa.streamlit.app/](https://agente-rag-ia-normativa.streamlit.app/)

---

## Descripción del Dominio y Justificación del Experto

El rápido despliegue de sistemas de Inteligencia Artificial Generativa y, de manera específica, de los **agentes autónomos (IA Agéntica)** —sistemas capaces de razonar, planificar, invocar herramientas externas y procesar datos personales de forma continua sin intervención humana directa— ha generado un reto normativo y de cumplimiento legal sin precedentes.

Este proyecto industrializa un **Consultor Legal Senior y Arquitecto de Cumplimiento Normativo en Inteligencia Artificial**, fundamentado en un motor RAG (*Retrieval-Augmented Generation*) conversacional y modular. El agente es capaz de auditar, interpretar e inferir riesgos técnico-legales combinando la capacidad de razonamiento de modelos de lenguaje avanzados con una base de conocimiento vectorial especializada en formato Markdown (`.md`).

### Corpus Normativo Integrado (`data/raw/`)
La base de conocimiento supera con holgura los requisitos de volumen de evaluación, delimitada en tres pilares legales y técnicos específicos:
* `reglamentoRGPD.md`: Texto estructurado en Markdown del **Reglamento General de Protección de Datos (RGPD)** de la Unión Europea, abarcando principios de licitud, consentimiento, bases jurídicas, derechos de los interesados y régimen de sanciones.
* `Orientaciones Ia Agéntica.md`: Guía técnica integral sobre **Inteligencia Artificial Agéntica y Privacidad**, que aborda los riesgos inherentes en el diseño arquitectónico de agentes autónomos (patrones *Chain of Thought*, bucles *ReAct*, delegación inter-agente) y sus medidas de mitigación requeridas.
* `EDPB_Opinion_2024_28.md`: Dictamen oficial en inglés (**Opinion 2024/28**) del **Comité Europeo de Protección de Datos (EDPB)** relativo al tratamiento de datos personales en el desarrollo, entrenamiento y despliegue de modelos de inteligencia artificial (*AI Models*).

---

## Arquitectura del Cuaderno y Control de Identificación (`_4.ipynb`)

El corazón analítico y de validación académica del proyecto reside en el cuaderno `notebooks/proyecto_final_agente_rag._4.ipynb`. Para garantizar una trazabilidad e integridad absolutas:
* **Celda Alumno (`uuid_alumno`):** El cuaderno inicia en la primera celda con el identificador de evaluación asignado (`uuid_alumno = "ff3f09cc-529b-4011-a745-256ac2565010"`).
* **Control de Entorno Defensivo:** Comprueba la disponibilidad de `GEMINI_API_KEY` antes de inicializar conexiones con ChromaDB o Google AI Studio.
* **Versión Estática Interactiva:** Para facilitar la revisión sin necesidad de levantar un kernel, el cuaderno evaluado se encuentra exportado a formato HTML tanto en `docs/proyecto_final_agente_rag._4.html` como en `notebooks/proyecto_final_agente_rag._4.html`.

---

## Segmentación Semántica en Dos Etapas (`Two-Stage Chunking`)

A diferencia de los pipelines tradicionales que trocean documentos en formato PDF mediante divisiones arbitrarias por número de caracteres (rompiendo artículos a la mitad y perdiendo el contexto legal), este proyecto diseña una tubería de datos orientada nativamente a Markdown:

1. **Etapa Estructural (`MarkdownHeaderTextSplitter`):** El procesador reconoce la jerarquía legal nativa del documento mediante cabeceras (`# Titulo_Ley`, `## Capitulo`, `### Articulo`). Corta el texto por las fronteras normativas y adhiere esas etiquetas a los metadatos semánticos.
2. **Etapa de Control de Granularidad (`RecursiveCharacterTextSplitter`):** Para evitar que los artículos extensos diluyan la precisión del embedding en el espacio vectorial, los bloques de la Etapa 1 se subdividen a un máximo de 1.000 caracteres con 150 de solapamiento (`overlap`). Cada fragmento resultante conserva intacta la genealogía normacional (`Titulo -> Capitulo -> Articulo`) en su metadato de origen.

---

## Justificación e Ingeniería del System Prompt (`template_rag`)

El comportamiento del consultor está gobernado por un System Prompt Maestro estructurado con reglas de blindaje, desduplicación y síntesis elástica:

### Perfil y Tono
El agente opera bajo el perfil de un **Consultor Legal Senior y Arquitecto en Privacidad de IA**. Su tono es estrictamente formal, analítico, objetivo e impersonal.

### Los 5 Guardrails Defensivos y Estratégicos
1. **Veracidad y Síntesis Elástica:** El agente responde únicamente basándose en los fragmentos recuperados de `corpus_normativo_v3`. Si el usuario plantea una consulta teórica o general sobre el dominio (RGPD o IA Agéntica), el agente integra y sintetiza de forma estructurada los conceptos aplicables presentes en el contexto, aclarando explícitamente qué aspectos normativos aporta la base de conocimientos sin incurrir en alucinaciones fuera del corpus.
2. **Memoria y Coherencia Multi-turno:** Instrucción explícita para consultar el historial reciente del chat (`messages`) y resolver pronombres, elipsis o referencias implícitas (*"¿Y qué multas se aplican en ese caso?"*).
3. **Guardrail de Dominio:** Si la consulta del usuario es ajena a la tecnología, la inteligencia artificial o el derecho (por ejemplo, preguntas sobre gastronomía, deportes o entretenimiento), el enrutador o el guardrail interceptan la petición y responden invariablemente:  
   *"No estoy entrenado para responder sobre ese tema"*.
4. **Guardrail de Ausencia de Información:** Si la pregunta pertenece al ámbito legal o tecnológico pero el precepto consultado no figura en los textos recuperados de `data/raw`, el agente reconoce el límite técnico con honestidad:  
   *"La información disponible en la base de conocimientos no permite responder a esta consulta"*.
5. **Guardrail de Idioma (Espejo Lingüístico Impermeable):** Para eliminar cualquier anclaje del modelo (*prompt anchoring*), el prompt maestro y los nodos de cortesía dividen estructuralmente sus instrucciones y cabeceras según el idioma detectado:
   * **Consulta en Español:** Redacta el 100% en español bajo las cabeceras `### Conclusión Directa`, `### Análisis Normativo` y `### Trazabilidad Jurídica`.
   * **Consulta en Inglés:** Redacta el 100% en inglés puro bajo las cabeceras `### Direct Conclusion`, `### Normative Analysis` y `### Legal Traceability`, traduciendo de forma técnica cualquier norma en español del corpus.

### Formato de Salida y Regla de Desduplicación de Citas
* **Conclusión Directa / Direct Conclusion:** Una frase concisa y categórica que responde al dictamen.
* **Análisis Normativo / Normative Analysis:** Explicación técnica fundamentada en los artículos consultados.
* **Trazabilidad Jurídica (Con Desduplicación Obligatoria):** Lista de fuentes bajo la sintaxis `Archivo (Título -> Capítulo -> Artículo)`. Si la búsqueda MMR devuelve varios fragmentos pertenecientes al exacto mismo artículo y sección, el agente **agrupa las referencias y escribe esa cabecera una sola vez** en la lista final.

---

## Orquestación Conversacional por Grafo V4 (`LangGraph`)

La versión modular V4 (`src/rag_engine.py` y `_4.ipynb`) orquesta un grafo `StateGraph` avanzado con clasificación condicional temprana y búsqueda híbrida con máxima diversidad:

* **Nodo `router` (Enrutador de Intención):** Clasificador conversacional de bajo coste computacional que evalúa la pregunta del usuario en tiempo real antes de consultar la base de datos, enrutándola hacia tres posibles trayectorias:
  * `LEGAL_RAG`: Consultas normativas, técnicas o de privacidad.
  * `GENERAL_LLM`: Saludos o fórmulas de cortesía conversacional.
  * `OUT_OF_DOMAIN`: Preguntas ajenas al dominio (deportes, cocina, entretenimiento).
* **Nodos Condicionales Rápidos (`direct_llm` / `out_of_domain`):** Si la intención es de cortesía o fuera de dominio, estos nodos responden de forma directa en el exacto idioma de la consulta (`Espejo Lingüístico bilingüe`), evitando llamadas innecesarias al motor de embeddings y reduciendo la latencia a menos de 1.5 segundos.
* **Nodo `rewrite_query` (Reescritura de Anáforas):** Si la consulta es `LEGAL_RAG`, analiza el historial reciente para reformular el turno actual en una pregunta autocontenida y autónoma.
* **Nodo `retrieve` (Búsqueda Híbrida MMR):** Invoca el índice local `ChromaDB` (`corpus_normativo_v3`) aplicando **Maximal Marginal Relevance (MMR)** (`top_k=4`, `fetch_k=20`, `lambda_mult=0.7`). Este algoritmo selecciona los fragmentos con mayor relevancia semántica pero maximizando la diversidad entre ellos, impidiendo que fragmentos adyacentes idénticos acaparen la ventana de contexto.
* **Nodo `generate` (Inferencia Legal RAG):** Ensambla el dictamen jurídico aplicando el prompt maestro bilingüe con `gemini-3.1-flash-lite` (temperatura `0.2`).
* **Persistencia por Hilo (`MemorySaver`):** El grafo se compila integrando el *checkpointer* oficial `MemorySaver` de LangGraph, gestionando el estado y la memoria conversacional de forma segura mediante identificadores de hilo (`thread_id`) independientes por usuario o sesión.

---

## Industrialización y Despliegue en Producción (Streamlit Cloud)

El sistema RAG ha sido completamente desacoplado del entorno de Jupyter y puesto en producción como una aplicación web modular y concurrente en Streamlit Cloud:

* **Enlace Público Oficial:** [https://agente-rag-ia-normativa.streamlit.app/](https://agente-rag-ia-normativa.streamlit.app/)
* **Motor Modular (`src/rag_engine.py`):** Encapsula la conexión con ChromaDB, la instanciación de Gemini y la compilación de la arquitectura V4 del grafo. Incluye parches de compatibilidad automáticos para servidores Linux en la nube (`pysqlite3`).
* **Interfaz Visual (`app.py`):**
  * **Trazabilidad en Vivo (`st.status`):** Visualiza en un panel expansible el recorrido técnico del grafo por cada consulta (clasificación del enrutador, reformulación, búsqueda MMR e inferencia del dictamen con tiempo en segundos).
  * **Aislamiento de Sesión (`uuid.uuid4`):** Cada usuario que abre la aplicación web o reinicia el historial recibe un identificador único que se inyecta en la invocación del motor (`config={"configurable": {"thread_id": st.session_state.thread_id}}`), garantizando la estricta separación de memorias entre usuarios en concurrencia.
  * **Gobernanza y Parámetros en Sidebar:** Monitorización en tiempo real de fragmentos cargados en RAM, temperatura del modelo e información de los 5 guardrails activos.

---

## Suite de Evaluación Automatizada (`_4.ipynb`)

El cuaderno principal ejecuta un banco de 7 pruebas de esfuerzo técnicas que verifican la robustez arquitectónica de todo el sistema en sus tres rutas del enrutador:
1. **Prueba 1 (Consulta Directa RGPD):** Licitud del tratamiento y condiciones del consentimiento (`LEGAL_RAG`).
2. **Prueba 2 (Razonamiento Multi-turno y Anáfora):** Sanciones aplicables al caso anterior sin mencionar explícitamente "RGPD" (`LEGAL_RAG` -> `rewrite_query`).
3. **Prueba 3 (IA Agéntica y Chain of Thought):** Auditoría técnica y riesgos de privacidad en bucles de agentes autónomos (`LEGAL_RAG`).
4. **Prueba 4 (Consulta en Inglés sobre EDPB):** Verificación del Espejo Lingüístico impermeable respondiendo un dictamen 100% en inglés con traducción técnica rigurosa del corpus en español (`LEGAL_RAG`).
5. **Prueba 5 (Guardrail Fuera de Dominio):** Rechazo inmediato ante una consulta sobre recetas gastronómicas (`OUT_OF_DOMAIN`).
6. **Prueba 6 (Guardrail de Ausencia de Información):** Consulta sobre drones agrícolas en Australia, reconociendo con honestidad los límites de la base vectorial local (`LEGAL_RAG`).
7. **Prueba 7 (Enrutador de Cortesía / Saludo):** Interceptación de un saludo en español o inglés respondiendo de forma conversacional (`GENERAL_LLM`).

Además, la celda final del cuaderno incluye el **Modo Interactivo en Vivo** (`while True:` con `input`), inyectando el `thread_id` en `MemorySaver` para permitir auditorías manuales ilimitadas desde el propio Jupyter.

---

## Requisitos e Instrucciones de Ejecución Local

### Prerrequisitos
* **Python:** 3.10 o superior (Verificado en entorno Python 3.14).
* **Clave API de Google Gemini (`GEMINI_API_KEY`):** Credencial activa de Google AI Studio configurada como variable de entorno.

### Instalación de Dependencias
En la terminal de comandos situada en el directorio raíz del repositorio:
```bash
pip install -r requirements.txt
```

### Configuración del Entorno (`.env`)
Crea un archivo llamado `.env` en la raíz del proyecto con tu clave secreta:
```env
GEMINI_API_KEY="AIzaSyTu_Clave_Secreta_De_Google_Gemini_Aqui"
```

### Ejecución de la Interfaz Web (Streamlit)
Lanza la interfaz gráfica local ejecutando:
```bash
streamlit run app.py
```

### Ejecución Paso a Paso del Cuaderno Jupyter
1. Abre tu terminal e inicia Jupyter:
   ```bash
   jupyter notebook
   ```
2. Abre `notebooks/proyecto_final_agente_rag._4.ipynb`.
3. Ejecuta las celdas en orden secuencial (`Shift + Enter`) para cargar el índice `vector_db/`, compilar el grafo LangGraph V4, observar el informe de las 7 pruebas de esfuerzo y abrir el chat interactivo en la celda final.

---

## Estructura Limpia del Repositorio Público

```text
├── app.py                                 # Interfaz Web interactiva en Streamlit (Despliegue en Cloud)
├── data/
│   └── raw/
│       ├── EDPB_Opinion_2024_28.md        # Dictamen EDPB 2024/28 sobre Modelos de IA (Inglés)
│       ├── Orientaciones Ia Agéntica.md   # Guía Técnica de IA Agéntica y Privacidad (Español)
│       └── reglamentoRGPD.md            # Texto normativo estructurado del RGPD (Español)
├── docs/
│   ├── build_notebook_v4.py               # Generador automatizado del Cuaderno V4
│   └── proyecto_final_agente_rag._4.html  # Versión estática HTML con resultados evaluados
├── notebooks/
│   ├── proyecto_final_agente_rag._4.ipynb # Cuaderno Jupyter V4 (Evaluación y Chat Interactivo)
│   └── proyecto_final_agente_rag._4.html  # Exportación HTML interactiva
├── src/
│   ├── __init__.py                        # Paquete modular
│   └── rag_engine.py                      # Motor RAG LangGraph V4 (Router + MMR + MemorySaver)
├── vector_db/
│   └── chroma.sqlite3                     # Base vectorial local pre-indexada (corpus_normativo_v3)
├── README.md                              # Documentación oficial del repositorio y despliegue
└── requirements.txt                       # Dependencias unificadas para producción y desarrollo
```

---

**Licencia y Créditos:** Proyecto desarrollado e industrializado para la evaluación final de Inteligencia Artificial Generativa y Agentes RAG dentro del Máster de Formación Permanente en Inteligencia Artificial y Ciencia de Datos. Todos los derechos reservados.

