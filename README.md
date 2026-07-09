# Sistema RAG Conversacional sobre Privacidad e Inteligencia Artificial Agéntica

**Repositorio Oficial — Proyecto Final de Máster en Inteligencia Artificial y Ciencia de Datos**  
**Dominio de Especialización:** Derecho Tecnológico, Cumplimiento Normativo (RGPD) y Gobernanza de Modelos de Inteligencia Artificial Agéntica  
**Pila Tecnológica:** Python 3.10+, LangChain, LangGraph, Google Gemini API, ChromaDB y Segmentación Semántica Jerárquica sobre Markdown

---

## Descripción del Dominio y Justificación del Experto

El rápido despliegue de sistemas de Inteligencia Artificial Generativa y, de manera específica, de los **agentes autónomos (IA Agéntica)** —sistemas capaces de razonar, planificar, invocar herramientas externas y procesar datos personales de forma continua sin intervención humana directa— ha generado un reto normativo y de cumplimiento legal sin precedentes.

Este proyecto industrializa un **Consultor Legal Senior y Arquitecto de Cumplimiento Normativo en Inteligencia Artificial**, fundamentado en un motor RAG (*Retrieval-Augmented Generation*) conversacional y modular. El agente es capaz de auditar, interpretar e inferir riesgos técnico-legales combinando la capacidad de razonamiento de modelos de lenguaje avanzados con una base de conocimiento vectorial especializada en formato Markdown (`.md`).

### Corpus Normativo Integrado (`data/raw/`)
La base de conocimiento supera con holgura los requisitos de volumen de evaluación (~20 páginas / 3 documentos mínimos), incorporando más de 200.000 caracteres distribuidos en tres textos legales y orientaciones institucionales:
* `reglamentoRGPD.md`: Texto estructurado en Markdown del **Reglamento General de Protección de Datos (RGPD)** de la Unión Europea, abarcando principios de licitud, consentimiento, bases jurídicas, derechos de los interesados y régimen de sanciones.
* `Orientaciones Ia Agéntica.md`: Guía técnica integral sobre **Inteligencia Artificial Agéntica y Privacidad**, que aborda los riesgos inherentes en el diseño arquitectónico de agentes autónomos (patrones *Chain of Thought*, bucles *ReAct*, delegación inter-agente) y sus medidas de mitigación requeridas.
* `EDPB_Opinion_2024_28.md`: Dictamen oficial en inglés (**Opinion 2024/28**) del **Comité Europeo de Protección de Datos (EDPB)** relativas al tratamiento de datos personales en el desarrollo, entrenamiento y despliegue de modelos de inteligencia artificial (*AI Models*).

---

## Arquitectura del Cuaderno y Control de Identificación (`_3.ipynb`)

El corazón del proyecto se ejecuta de forma autónoma desde el cuaderno principal situado en `notebooks/proyecto_final_agente_rag._3.ipynb`. Para garantizar una compatibilidad y trazabilidad académicas absolutas:
* **Celda Alumno (`uuid_alumno`):** El cuaderno inicia exactamente en la primera celda con el identificador de evaluación asignado (`uuid_alumno = "ff3f09cc-529b-4011-a745-256ac2565010"`), permitiendo la verificación y auditoría automática al arrancar el kernel de Python.
* **Control de Entorno Defensivo:** Antes de iniciar procesos de vectorización, el código comprueba activamente la disponibilidad de la variable `GEMINI_API_KEY` en el entorno, deteniendo la ejecución con un mensaje claro si no se detectan credenciales.

---

## Segmentación Semántica en Dos Etapas (`Two-Stage Chunking`)

A diferencia de los pipelines tradicionales que trocean documentos en formato PDF mediante divisiones arbitrarias por número de caracteres (rompiendo artículos a la mitad y perdiendo el contexto legal), este proyecto diseña una tubería de datos orientada nativamente a Markdown:

1. **Etapa Estructural (`MarkdownHeaderTextSplitter`):** El procesador reconoce la jerarquía legal nativa del documento mediante cabeceras (`# Titulo_Ley`, `## Capitulo`, `### Articulo`). Corta el texto exactamente por las fronteras normativas y adhiere esas etiquetas a los metadatos semánticos.
2. **Etapa de Control de Granularidad (`RecursiveCharacterTextSplitter`):** Para evitar que los artículos extensos diluyan la precisión del embedding en el espacio vectorial, los bloques de la Etapa 1 se subdividen a un máximo de 1.000 caracteres con 150 de solapamiento (`overlap`). Cada fragmento resultante conserva intacta la genealogía normacional (`Titulo -> Capitulo -> Articulo`) en su metadato de origen.

---

## Justificación e Ingeniería del System Prompt (`template_rag`)

El comportamiento del consultor está gobernado estrictamente por el System Prompt Maestro documentado e implementado en el código, diseñado para garantizar rigor jurídico, neutralidad analítica y renderizado progresivo en Markdown.

### Perfil y Tono
El agente opera bajo el perfil de un **Consultor Legal Senior y Arquitecto en Privacidad de IA**. Su tono es estrictamente formal, analítico, objetivo e impersonal. No emite opiniones subjetivas, especulaciones ni asume premisas no respaldadas por la ley.

### Los 5 Guardrails Defensivos y Estratégicos
El prompt implementa cinco barreras explícitas de protección y alineamiento:
1. **Veracidad y Cero Alucinaciones:** El agente tiene prohibido inventar o inferir legislación que no figure de forma explícita en los fragmentos de contexto recuperados por el buscador.
2. **Memoria y Coherencia Multi-turno:** Instrucción explícita para consultar el historial reciente del chat (`messages`) para resolver pronombres, elipsis o referencias implícitas (*"¿Y qué multas se aplican en ese caso?"*).
3. **Guardrail de Dominio:** Si la consulta del usuario es ajena a la tecnología, la inteligencia artificial o el derecho (por ejemplo, preguntas sobre gastronomía, deportes o entretenimiento), el agente intercepta la petición y responde invariablemente:  
   *"No estoy entrenado para responder sobre ese tema"*.
4. **Guardrail de Ausencia de Información:** Si la pregunta pertenece al ámbito legal o tecnológico pero el precepto consultado no figura en los textos recuperados, el agente reconoce el límite técnico con honestidad:  
   *"La información disponible en la base de conocimientos no permite responder a esta consulta"*.
5. **Guardrail de Idioma (Espejo Lingüístico Obligatorio):** Detecta automáticamente el idioma en que está escrita la pregunta actual y redacta el dictamen siempre en ese exacto mismo idioma. Si la consulta es en español, responde el 100% en español; si es en inglés, redacta el 100% en inglés traduciendo los artículos jurídicos recuperados.

### Formato de Salida y Regla de Desduplicación de Citas
Para permitir una visualización limpia al usar `display(Markdown(...))` en Jupyter, la respuesta se articula siempre en tres bloques encabezados sin numeración artificial:
* **Conclusión Directa:** Una frase ejecutiva resolutiva respondiendo a la consulta.
* **Análisis Normativo:** Explicación técnica detallada en viñetas o párrafos concisos fundamentada en los artículos consultados.
* **Trazabilidad Jurídica (Con Desduplicación Obligatoria):** Lista de fuentes normativas invocadas bajo la sintaxis `Archivo (Título -> Capítulo -> Artículo)`. Para evitar saturación visual, si el buscador vectorial recupera múltiples fragmentos de un mismo artículo legal, el prompt obliga al modelo a **agrupar y unificar la referencia en una sola viñeta legal**.

---

## Orquestación Conversacional por Grafo de Nodos (`LangGraph`)

El motor del agente se estructura con **LangGraph** (`StateGraph`) mediante tres nodos especializados conectados en un flujo estrictamente secuencial y determinista, optimizando la latencia y la resiliencia técnica:

* **Nodo `rewrite_query` (Reescritura de Anáforas):** Inspecciona el historial conversacional (`messages`). Si el turno actual depende de un intercambio anterior, invoca al LLM para reformular la consulta en una pregunta autónoma y auto-explicativa antes de consultar el índice.
* **Nodo `retrieve` (Búsqueda Vectorial Defensiva):** Consulta el índice `ChromaDB` local y recupera los 4 fragmentos legales con mayor similitud semántica (`top_k=4`). Incluye una interceptación genérica de excepciones para blindar la ejecución académica del grafo; en un paso a producción se segmentarían errores de API de red frente a fallos de I/O en disco local.
* **Nodo `generate` (Inferencia Legal con Ventana Deslizante):** Recibe el contexto recuperado, la consulta reformulada e inyecta el historial conversacional empleando una **Ventana Deslizante de Memoria (`Sliding Window`)** que retiene únicamente los últimos 4 turnos (2 intercambios). Esto previene la saturación de tokens e impide que idiomas o temas del historial remoto interfieran en el turno actual, generando el dictamen final utilizando el modelo **`gemini-3.1-flash-lite`** a **temperatura 0.2**.

---

## Resiliencia en Disco y Gestión de Cuotas de API (`Batching + Backoff`)

Uno de los principales retos de ingeniería al operar con modelos en la nube (capa gratuita de Google AI Studio) es la gestión de límites de peticiones por minuto (`Error 429 Rate Limit`). El código implementa un blindaje en tres niveles:
* **Persistencia Local en Disco (`vector_db/`):** Todos los vectores construidos con `models/gemini-embedding-001` se almacenan físicamente en el directorio `vector_db/` bajo el identificador `corpus_normativo_v3`.
* **Indexación Condicional Inteligente:** Al ejecutar la sección de base de datos, el código verifica si el índice ya cuenta con vectores guardados (`vectores_existentes > 0`). Si existen y la bandera `forzar_reindexacion` es `False`, el bucle de embedding se omite de forma instantánea, cargando el conocimiento en milisegundos sin consumir peticiones de API.
* **Lotes Controlados y Pausa Exponencial (`Backoff`):** Cuando la base debe regenerarse, los 293 fragmentos se indexan en bloques controlados de 15 trozos con pausas de 2 segundos inter-lote. Si el servidor de Google devuelve una saturación temporal (Error 429), el sistema intercepta la excepción e **inyecta una pausa de protección de 65 segundos**, reiniciando el contador móvil de Google y reanudando la indexación hasta completar el 100% sin cuelgues del kernel.

---

## Suite de Evaluación Automatizada y Chat Interactivo (`_3.ipynb`)

El cuaderno final no solo define las funciones y el grafo, sino que ejecuta de manera automatizada y documentada **dos métodos de prueba e interacción**:

### 1. Suite de Evaluación Automatizada (6 Pruebas de Esfuerzo)
El código ejecuta un banco de pruebas técnicas que audita el comportamiento real del agente en todos sus frentes, renderizando cada resultado en pantalla mediante `display(Markdown(...))`:
* **Prueba 1 (Consulta Directa RGPD):** Evaluación sobre condiciones de licitud del tratamiento de datos y rol del consentimiento.
* **Prueba 2 (Razonamiento Multi-turno y Anáfora):** Comprobación de memoria conversacional consultando por el régimen de sanciones aplicable al caso de la Prueba 1 sin mencionar explícitamente la palabra "RGPD".
* **Prueba 3 (IA Agéntica y Chain of Thought):** Evaluación del dominio legal en arquitecturas complejas de agentes autónomos y riesgos de privacidad.
* **Prueba 4 (Consulta en Inglés y Traducción):** Evaluación del Dictamen EDPB solicitando un análisis técnico en inglés, verificando la respuesta en el idioma de origen y la traducción rigurosa de normas en español.
* **Prueba 5 (Guardrail Fuera de Dominio):** Intento de desvío conversacional preguntando por la receta clásica de la paella valenciana. Verificación de rechazo del agente.
* **Prueba 6 (Guardrail de Ausencia de Información):** Consulta sobre legislación específica de drones agrícolas en Australia. Verificación de reconocimiento de límites de su base vectorial.

### 2. Modo Interactivo en Vivo (Celda de Chat Conversacional)
Como cierre práctico del proyecto (Paso 6 del Enunciado), la última celda del cuaderno implementa un bucle `while True:` con captura de entrada de teclado (`input`). Al ejecutarla, el usuario puede dialogar libremente en tiempo real con el consultor legal RAG dentro de Jupyter, encadenando preguntas complejas con memoria conversacional y recibiendo respuestas instantáneas perfectamente estructuradas en Markdown. Para finalizar la sesión interactiva, basta con escribir las palabras clave `salir` o `terminar`.

---

## Requisitos e Instrucciones de Ejecución

### Prerrequisitos
* **Python:** 3.10 o superior (Verificado en entorno Python 3.14).
* **Clave API de Google Gemini (`GEMINI_API_KEY`):** Credencial activa de Google AI Studio configurada como variable de entorno.

### Instalación de Dependencias (`requirements.txt`)
En la terminal de comandos situada en el directorio raíz del repositorio, ejecuta la instalación unificada:
```bash
pip install -r requirements.txt
```

### Configuración del Entorno (`.env`)
Crea un archivo de texto llamado `.env` en la raíz del proyecto (este archivo está excluido por defecto por motivos de seguridad en `.gitignore`) con el siguiente contenido:
```env
GEMINI_API_KEY="AIzaSyTu_Clave_Secreta_De_Google_Gemini_Aqui"
```

### Ejecución Paso a Paso del Cuaderno
1. Abre tu terminal en la raíz del repositorio e inicia el servidor de cuadernos:
   ```bash
   jupyter notebook
   ```
2. Navega al directorio `notebooks/` y abre el archivo:
   `proyecto_final_agente_rag._3.ipynb`
3. Ejecuta las celdas en orden secuencial (`Shift + Enter`). El código cargará los datos desde `../data/raw/`, detectará la base local en `../vector_db/`, compilará el grafo LangGraph, ejecutará el reporte completo de las 6 pruebas de evaluación y abrirá el chat interactivo en la celda final.

---

## Estructura Limpia del Repositorio Público

```text
├── data/
│   └── raw/
│       ├── EDPB_Opinion_2024_28.md          # Dictamen EDPB sobre IA (Inglés)
│       ├── Orientaciones Ia Agéntica.md     # Guía Técnica de IA Agéntica y Privacidad (Español)
│       └── reglamentoRGPD.md              # Texto normativo estructurado del RGPD (Español)
├── notebooks/
│   └── proyecto_final_agente_rag._3.ipynb   # Cuaderno Jupyter de Evaluación (Con celda uuid_alumno y chat en vivo)
├── vector_db/
│   └── chroma.sqlite3                     # Base vectorial con los 903 fragmentos semánticos pre-indexados
├── .gitignore                             # Bloqueo de seguridad para .env, .venv, y carpeta local docs/
├── README.md                              # Documentación oficial del repositorio (Dominio, arquitectura y uso)
└── requirements.txt                       # Dependencias oficiales para la ejecución y evaluación
```

---

**Licencia y Créditos:** Proyecto desarrollado e industrializado para la evaluación final de Inteligencia Artificial Generativa y Agentes RAG dentro del Máster de Formación Permanente en Inteligencia Artificial y Ciencia de Datos. Todos los derechos reservados.
