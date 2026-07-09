# Sistema RAG Conversacional sobre Regulación e Inteligencia Artificial Agéntica

**Repositorio del Proyecto Final de Máster — Inteligencia Artificial Generativa y Agentes Autónomos**  
**Dominio de Conocimiento:** Derecho Tecnológico, Cumplimiento Normativo (RGPD) y Gobernanza de Modelos de Inteligencia Artificial Agéntica  
**Pila Tecnológica:** Python, LangChain, LangGraph, Google Gemini API, ChromaDB y Segmentación Semántica en Markdown

---

## Descripción del Dominio del Experto

El rápido despliegue de sistemas de Inteligencia Artificial Generativa y, en particular, de **agentes autónomos (IA Agéntica)** —capaces de planificar, ejecutar herramientas, tomar decisiones complejas y procesar datos personales sin supervisión humana constante— ha generado un importante desafío en materia de cumplimiento normativo y privacidad de datos.

Este proyecto implementa un **Consultor Legal Senior y Arquitecto de Cumplimiento Normativo en Inteligencia Artificial** fundamentado en un motor RAG (*Retrieval-Augmented Generation*) conversacional y modular. El agente es capaz de auditar, responder e inferir riesgos técnico-legales combinando el razonamiento formal de modelos de lenguaje avanzados con una base de conocimiento vectorial especializada en formato Markdown (`.md`).

### Corpus Normativo Integrado
La base de conocimiento local supera con creces el requisito de volumen del proyecto (~20 páginas / 3 documentos mínimos), incorporando más de 200.000 caracteres distribuidos en tres pilares normativos e institucionales:
* `reglamentoRGPD.md`: Texto estructurado en Markdown del **Reglamento General de Protección de Datos (RGPD)** de la Unión Europea, abarcando principios de licitud, consentimiento, bases jurídicas, derechos de los interesados y régimen de sanciones.
* `Orientaciones Ia Agéntica.md`: Guía práctica integral en español sobre **Inteligencia Artificial Agéntica y Privacidad**, que aborda los riesgos en el diseño arquitectónico de agentes (patrones como *Chain of Thought*, bucles *ReAct*, delegación inter-agente) y sus medidas de mitigación.
* `EDPB_Opinion_2024_28.md`: Dictamen oficial en inglés (**Opinion 2024/28**) del **Comité Europeo de Protección de Datos (EDPB)** sobre el tratamiento de datos personales en el entrenamiento y despliegue de modelos de inteligencia artificial (AI Models).

---

## Arquitectura Técnica y Segmentación Jerárquica

A diferencia de los enfoques tradicionales que ingieren documentos en formato PDF mediante divisiones arbitrarias por número de caracteres, este proyecto diseña e industrializa una **tubería de datos orientada a Markdown puro (Two-Stage Semantic Chunking)**:

### Segmentación en Dos Etapas (`Two-Stage Chunking`)
1. **Etapa Estructural (`MarkdownHeaderTextSplitter`):** El procesador reconoce la jerarquía legal nativa del documento mediante cabeceras (`# Titulo_Ley`, `## Capitulo`, `### Articulo`). Corta el documento exactamente por las fronteras normativas y transfiere esas etiquetas a los metadatos vectoriales.
2. **Etapa de Control de Tamaño (`RecursiveCharacterTextSplitter`):** Para evitar que los artículos excesivamente largos (como los preceptos sancionadores) diluyan la precisión del embedding semántico, los bloques de la Etapa 1 se subdividen a un máximo de 1.000 caracteres con 150 de solapamiento (`overlap`). La segunda etapa conserva intacta la genealogía legal (`Titulo -> Capitulo -> Articulo`) en el metadato de cada fragmento.

---

## Justificación e Ingeniería del System Prompt

El comportamiento del agente está gobernado por el System Prompt Maestro (`template_rag`), diseñado específicamente para garantizar la rigurosidad jurídica, evitar alucinaciones y mantener una presentación profesional renderizable en Markdown.

### Perfil y Tono del Agente
El modelo actúa como un **Consultor Legal Senior**. Se le exige un registro formal, analítico, riguroso y estrictamente acotado al conocimiento normativo facilitado. No emite opiniones subjetivas ni suposiciones no respaldadas por la ley.

### Los 5 Guardrails de Seguridad y Comportamiento Estratégico
El prompt incorpora cinco barreras de protección de obligado cumplimiento:
1. **Veracidad y Cero Alucinaciones:** El agente tiene terminantemente prohibido inventar o suponer normas fuera de los fragmentos recuperados en el bloque de contexto.
2. **Memoria y Coherencia Multi-turno:** Instrucción explícita para consultar el historial reciente ante pronombres, elipsis o referencias implícitas del usuario (*"¿Y qué multas tiene eso?"*).
3. **Guardrail de Dominio:** Si la pregunta es ajena a la tecnología, la inteligencia artificial o el derecho (ejemplo: consultas sobre cocina, deportes o entretenimiento), el agente intercepta la petición y responde estrictamente:  
   *"No estoy entrenado para responder sobre ese tema"*.
4. **Guardrail de Ausencia de Información:** Si la pregunta pertenece al dominio legal o tecnológico pero el precepto concreto no figura en los textos recuperados, el agente responde con honestidad técnica:  
   *"La información disponible en la base de conocimientos no permite responder a esta consulta"*.
5. **Guardrail de Idioma Multilingüe y Traducción:** Detecta automáticamente el idioma de la consulta. Si el usuario pregunta en inglés sobre legislación en español, o viceversa, el agente redacta el 100% de su dictamen final en el idioma del usuario, traduciendo con exactitud jurídica los preceptos consultados.

### Formato de Salida y Regla de Desduplicación de Citas
Para facilitar la auditoría y su visualización en Jupyter Notebook mediante `display(Markdown(...))`, la respuesta se estructura en tres bloques diferenciados con encabezados sin numerar:
* **Conclusión Directa:** Una frase ejecutiva clara y categórica respondiendo a la pregunta.
* **Análisis Normativo:** Explicación técnica detallada en viñetas o párrafos breves fundamentada en los artículos leídos.
* **Trazabilidad Jurídica (Con Desduplicación Obligatoria):** Lista de fuentes normativas con la estructura `Archivo (Título -> Capítulo -> Artículo)`. Para evitar bloques redundantes, si el buscador lee varios fragmentos de un mismo artículo legal, el prompt obliga al modelo a **agrupar la referencia en una sola entrada**, manteniendo una presentación tipográfica limpia.

---

## Orquestación Conversacional con LangGraph (`StateGraph`)

El motor del agente se construye utilizando **LangGraph** mediante un grafo de estado (`GraphState`) compuesto por tres nodos secuenciales sin ciclos redundantes de autoevaluación (que elevarían innecesariamente la latencia y el consumo de cuota):

* **Nodo `rewrite_query` (Reescritura de Anáforas):** Analiza el historial conversacional (`messages`). Si el usuario realiza una pregunta que depende del contexto previo (*"¿Qué multas administrativas se aplican en ese caso?"*), el modelo LLM reformula la consulta de manera autónoma antes de consultar el índice.
* **Nodo `retrieve` (Búsqueda Vectorial Defensiva):** Consulta la base de datos `ChromaDB` (`vector_db`) y recupera los 4 fragmentos legales con mayor similitud del coseno (`top_k=4`). Incluye manejo de excepciones ante interrupciones locales.
* **Nodo `generate` (Razonamiento Legal y Guardrails):** Recibe el contexto recuperado, el historial del chat y la consulta del usuario, inyectando estos datos en el prompt `template_rag` para generar el dictamen utilizando el modelo **`gemini-2.5-flash-lite`** (configurado a **temperatura 0.2** para combinar naturalidad expositiva con determinismo legal).

---

## Resiliencia en Disco y Protección de Cuota de API (`Batching + Backoff`)

Uno de los principales retos prácticos al operar con modelos en la nube (Google AI Studio Free Tier) es la saturación temporal por límites de peticiones por minuto (`Error 429 Too Many Requests`). El proyecto resuelve este problema con una arquitectura defensiva:
* **Almacenamiento Vectorial Local (`vector_db/`):** Todos los vectores indexados con `models/gemini-embedding-001` se persisten en disco bajo la colección `corpus_normativo_v3`.
* **Indexación Condicional:** Al ejecutar el cuaderno, el script comprueba si la colección ya cuenta con vectores guardados (`vectores_existentes > 0`). Si ya existen y el parámetro `forzar_reindexacion` está en `False`, se omite el bucle de embedding, cargando el conocimiento en milisegundos sin consumir peticiones de la API.
* **Lotes Reducidos y Pausas de Seguridad (`Backoff`):** Cuando se crea la base por primera vez, los 293 trozos se envían en lotes controlados de 15 fragmentos con pausas inter-lote de 2 segundos. Si el servidor devuelve un error de saturación temporal (429), el código intercepta la excepción, informa por consola y **aplica una pausa automática de 65 segundos** para permitir que el contador móvil de Google se reinicie a cero, completando la indexación al 100% sin cuelgues ni pérdidas de datos.

---

## Requisitos y Configuración del Entorno

### Prerrequisitos del Sistema
* **Python:** Versión 3.10 o superior (Verificado en Python 3.14).
* **Clave de API de Google Gemini (`GEMINI_API_KEY`):** Es imprescindible disponer de una clave activa de Google AI Studio.

### Instalación de Dependencias
Abre tu terminal en el directorio raíz del repositorio e instala las librerías oficiales requeridas:

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters
pip install langchain-google-genai langchain-chroma chromadb langgraph
pip install ipython pypdf python-dotenv
```

### Configuración de la Clave API
Guarda tu clave secreta de Gemini creando un archivo de texto llamado `.env` en el directorio raíz del proyecto con el siguiente formato:

```env
GEMINI_API_KEY="AIzaSyTu_Clave_Secreta_De_Google_Gemini_Aqui"
```

*Nota de seguridad:* El archivo `.env` o la carpeta `vector_db/` (si generas tus propios vectores) se pueden excluir en `.gitignore` para no exponer credenciales en repositorios públicos.

---

## Instrucciones Paso a Paso para la Ejecución del Cuaderno

El proyecto está autocontenido dentro de la carpeta `notebooks/`. Para garantizar la máxima portabilidad, el código utiliza **rutas relativas retrocediendo un nivel (`..`)** para acceder a `data/raw/` y a `vector_db/`.

1. **Iniciar el Servidor de Jupyter Notebook:**
   Abra una terminal en la carpeta raíz del repositorio y ejecute:
   ```bash
   jupyter notebook
   ```
2. **Abrir el Cuaderno Final:**
   En la interfaz web de Jupyter, navegue a la subcarpeta `notebooks/` y abra el archivo:
   `proyecto_final_agente_rag._3.ipynb`
3. **Ejecución Secuencial de las Celdas:**
   Ejecute las celdas en orden desde la parte superior:
   * **Sección 1 y 2 (Entorno y API Key):** Comprueba la disponibilidad de las librerías y verifica silenciosamente que `GEMINI_API_KEY` esté cargada en memoria.
   * **Sección 3 (Ingesta Normativa):** Carga los 3 archivos Markdown desde `../data/raw/` e imprime una tabla de verificación de lectura y pesos en KB.
   * **Sección 4 (Troceado Semántico):** Aplica la segmentación en dos etapas e imprime una muestra de los metadatos jerárquicos (`Titulo_Ley`, `Capitulo`, `Articulo`).
   * **Sección 5 (ChromaDB Local):** Conecta con `../vector_db/chroma.sqlite3`. Si la base ya existe, informa su carga en memoria instantánea (`903 vectores`). Si no existe, inicia el bucle por lotes de 15 trozos con protección contra errores 429.
   * **Sección 6 (LangGraph):** Orquesta y compila el grafo conversacional en memoria.
   * **Sección 7 (Suite de Evaluación Automatizada):** Ejecuta una auditoría con **6 casos de prueba técnicos renderizados en Markdown**:
     1. Consulta directa sobre bases jurídicas en el RGPD.
     2. Prueba de memoria multi-turno con anáfora (*"¿Y qué multas administrativas se pueden imponer...?"*).
     3. Análisis técnico de riesgos en Inteligencia Artificial Agéntica (*Chain of Thought*).
     4. Consulta multilingüe en inglés sobre el Dictamen del EDPB (demostrando la traducción automática de conceptos y respuesta 100% en inglés).
     5. Activación del Guardrail fuera de dominio (rechazando una consulta gastronómica sobre la paella valenciana).
     6. Activación del Guardrail de ausencia de información (reconociendo el límite normativo al consultar sobre legislación de drones agrícolas en Australia).
   * **Sección 8 (Modo Interactivo en Vivo - Paso 6 del Enunciado):**  
     Ejecuta la última celda del cuaderno para activar el chat en directo. Se abrirá una caja de texto (`input`) directamente en Jupyter donde podrás dialogar libremente con el Consultor Legal RAG, haciendo preguntas de seguimiento con memoria conversacional y viendo respuestas instantáneas renderizadas en formato Markdown jerárquico. Para salir de la sesión interactiva, escribe `salir` o `terminar`.

---

## Estructura de Carpetas del Repositorio

```text
├── data/
│   └── raw/
│       ├── EDPB_Opinion_2024_28.md        # Dictamen EDPB sobre IA (Ingles)
│       ├── Orientaciones Ia Agéntica.md     # Guia Tecnica sobre Agentes e IA (Espanol)
│       └── reglamentoRGPD.md              # Texto estructurado del RGPD (Espanol)
├── notebooks/
│   └── proyecto_final_agente_rag._3.ipynb   # Cuaderno Jupyter Principal (Evaluacion + Chat)
├── vector_db/
│   └── chroma.sqlite3                     # Base de datos vectorial persistida en disco
├── .env                                   # Variable local con GEMINI_API_KEY (No incluir en git)
├── build_notebook_v3.py                   # Script de ingenieria que genera el cuaderno _3.ipynb
├── modificaciones.md                      # Registro de arquitectura, tono y control de calidad
├── proyecto_v3.md                         # Memoria tecnica original de practicas
└── README.md                              # Este documento oficial de documentacion del repositorio
```

---

**Licencia y Contacto:** Proyecto desarrollado con fines académicos e institucionales en el marco del Máster en Inteligencia Artificial y Ciencia de Datos. Todos los derechos reservados al autor.
