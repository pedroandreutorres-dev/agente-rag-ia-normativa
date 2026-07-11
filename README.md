# Consultor RAG: Protección de Datos (RGPD / EDPB) e Inteligencia Artificial Agéntica

**Repositorio Oficial — Proyecto Final de Máster en Inteligencia Artificial y Ciencia de Datos**  
**Dominio de Especialización:** Derecho Tecnológico, Cumplimiento Normativo (RGPD), Dictámenes del EDPB y Gobernanza de Sistemas de Inteligencia Artificial Agéntica  
**Pila Tecnológica:** Python 3.10+, LangChain, LangGraph (StateGraph + Router condicional), Google Gemini API (`gemini-3.1-flash-lite`), ChromaDB (MMR), MemorySaver y Streamlit  
**Aplicación Web en Vivo (Streamlit Cloud):** [https://agente-rag-ia-normativa.streamlit.app/](https://agente-rag-ia-normativa.streamlit.app/)

---

## Descripción del Dominio y Justificación Arquitectónica

El rápido despliegue de sistemas de Inteligencia Artificial Generativa y, de manera específica, de los **agentes autónomos (IA Agéntica)** —sistemas capaces de razonar, planificar, invocar herramientas externas y procesar datos personales de forma continua sin intervención humana directa— ha generado un reto normativo y de cumplimiento legal sin precedentes.

Este proyecto industrializa un **Consultor Legal Senior y Arquitecto de Cumplimiento Normativo en Inteligencia Artificial**, fundamentado en un motor RAG (*Retrieval-Augmented Generation*) conversacional y modular. El agente audita, interpreta e infiere riesgos técnico-legales combinando la capacidad de razonamiento de modelos de lenguaje avanzados con una base de conocimiento vectorial especializada en formato Markdown (`.md`).

### Corpus Normativo Integrado (`data/raw/`)
La base de conocimiento se estructura en tres pilares legales y técnicos fundamentales:
* `reglamentoRGPD.md`: Texto estructurado en Markdown del **Reglamento General de Protección de Datos (RGPD)** de la Unión Europea, abarcando principios de licitud, consentimiento, bases jurídicas, derechos de los interesados y régimen de sanciones.
* `Orientaciones Ia Agéntica.md`: Guía técnica integral sobre **Inteligencia Artificial Agéntica y Privacidad**, que aborda los riesgos inherentes en el diseño arquitectónico de agentes autónomos (patrones *Chain of Thought*, bucles *ReAct*, delegación inter-agente) y las medidas de mitigación requeridas.
* `EDPB_Opinion_2024_28.md`: Dictamen oficial en inglés (**Opinion 2024/28**) del **Comité Europeo de Protección de Datos (EDPB)** relativo al tratamiento de datos personales en el desarrollo, entrenamiento y despliegue de modelos de inteligencia artificial (*AI Models*).

---

## Arquitectura del Cuaderno y Humanización (`_5.ipynb`)

El análisis técnico, la justificación de ingeniería y la validación experimental del proyecto se encuentran en el cuaderno `notebooks/proyecto_final_agente_rag._5.ipynb`. 

Bajo la estrategia de humanización e ingeniería técnica (**Camino A**), toda la documentación, celdas Markdown y comentarios han sido redactados desde una perspectiva empírica, eliminando tecnicismos publicitarios de inteligencia artificial y fundamentando cada decisión arquitectónica con claridad y rigor:
* **Celda Alumno (`uuid_alumno`):** El cuaderno inicia con el identificador exacto de evaluación (`uuid_alumno = "ff3f09cc-529b-4011-a745-256ac2565010"`).
* **Control de Entorno Defensivo:** Verifica la presencia y validez de la clave `GEMINI_API_KEY` antes de establecer conexiones locales con ChromaDB o llamadas a la API de Google.
* **Visualización del Grafo:** La cabecera e inicialización incluyen el renderizado visual del diagrama de flujo del grafo mediante `app_rag.get_graph().draw_mermaid_png()`.
* **Exportación HTML Sincronizada:** El cuaderno ejecutado localmente con todas sus salidas y resultados de evaluación se encuentra exportado a formato HTML interactivo en `notebooks/proyecto_final_agente_rag._5.html`.

---

## Segmentación Semántica en Dos Etapas (`Two-Stage Chunking`)

Para garantizar que el modelo lingüístico reciba el contexto legal exacto sin fragmentar preceptos normativos, se implementa una tubería de preprocesamiento en dos etapas diseñada para formato Markdown:

1. **Etapa Estructural (`MarkdownHeaderTextSplitter`):** Reconoce la jerarquía legal nativa del documento mediante cabeceras (`# Titulo_Ley`, `## Capitulo`, `### Articulo`). Corta el texto preservando las fronteras normativas y adhiere la estructura jerárquica a los metadatos de cada bloque.
2. **Etapa de Control de Granularidad (`RecursiveCharacterTextSplitter`):** Para evitar que los artículos extensos diluyan la precisión del embedding en el espacio vectorial, los bloques de la Etapa 1 se subdividen a un máximo de 1.000 caracteres con 150 de solapamiento (`overlap`). Cada trozo conserva en su metadato de origen la trazabilidad jerárquica exacta (`Titulo_Ley -> Capitulo -> Articulo`).

---

## Justificación e Ingeniería del System Prompt (`template_rag`)

El comportamiento del consultor está gobernado por un Prompt Maestro estructurado con reglas de blindaje, desduplicación y síntesis elástica:

### Perfil y Tono
El agente opera bajo el perfil de un **Consultor Legal Senior y Arquitecto en Privacidad de IA**. Su tono es formal, analítico, riguroso y estrictamente fundamentado en las fuentes.

### Los 5 Guardrails Defensivos y Estratégicos
1. **Veracidad y Síntesis Elástica:** El agente responde basándose exclusivamente en los fragmentos recuperados del corpus. Si el usuario plantea una consulta conceptual o general sobre el dominio (RGPD o IA Agéntica), el agente integra y sintetiza de forma estructurada los conceptos aplicables presentes en el contexto, aclarando qué aspectos normativos aporta la base de conocimientos sin incurrir en alucinaciones fuera de ella.
2. **Memoria y Coherencia Multi-turno:** Instrucción para consultar el historial reciente del chat (`messages`) y resolver pronombres, elipsis o referencias implícitas en conversaciones prolongadas.
3. **Guardrail de Dominio:** Si la consulta del usuario es ajena a la tecnología, la inteligencia artificial o el derecho (preguntas sobre gastronomía, deportes o entretenimiento), la arquitectura intercepta la petición y responde invariablemente:  
   *"No estoy entrenado para responder sobre ese tema"*.
4. **Guardrail de Ausencia de Información:** Si la pregunta pertenece al ámbito legal o tecnológico pero el precepto consultado no figura en los textos recuperados del corpus, el agente reconoce el límite con honestidad:  
   *"La información disponible en la base de conocimientos no permite responder a esta consulta"*.
5. **Guardrail de Idioma (Espejo Lingüístico Impermeable):** Para evitar el anclaje lingüístico del modelo (*prompt anchoring*), el prompt maestro y los nodos condicionales dividen sus instrucciones y cabeceras según el idioma detectado:
   * **Consulta en Español:** Redacta el 100% en español bajo las cabeceras `### Conclusión Directa`, `### Análisis Normativo` y `### Trazabilidad Jurídica`.
   * **Consulta en Inglés:** Redacta el 100% en inglés bajo las cabeceras `### Direct Conclusion`, `### Normative Analysis` y `### Legal Traceability`, traduciendo con precisión técnica cualquier norma en español del corpus.

### Formato de Salida y Regla de Desduplicación de Citas
* **Conclusión Directa / Direct Conclusion:** Una frase concisa y categórica que responde al dictamen.
* **Análisis Normativo / Normative Analysis:** Explicación detallada en viñetas o párrafos breves basada en los artículos consultados.
* **Trazabilidad Jurídica (Con Desduplicación Obligatoria):** Lista de fuentes bajo la sintaxis `Archivo (Título -> Capítulo -> Artículo)`. Si la búsqueda MMR devuelve varios fragmentos pertenecientes exactamente al mismo artículo y sección, el agente **agrupa las referencias y escribe esa cabecera una sola vez** en la lista final.

---

## Orquestación Conversacional por Grafo (`LangGraph`)

El motor modular (`src/rag_engine.py`) y el cuaderno orquestan un `StateGraph` condicional equipado con el reductor incremental `add_messages` y búsqueda híbrida diversificada:

* **Nodo `router` (Enrutador de Intención):** Clasificador conversacional determinista (`temperature=0.0`) que evalúa la pregunta del usuario antes de consultar la base de datos, enrutándola hacia tres trayectorias:
  * `LEGAL_RAG`: Consultas normativas, jurídicas o sobre auditoría de IA.
  * `GENERAL_LLM`: Saludos o fórmulas de cortesía conversacional.
  * `OUT_OF_DOMAIN`: Preguntas ajenas al dominio normativo o tecnológico.
* **Nodos Condicionales Rápidos (`direct_llm` / `out_of_domain`):** Si la intención es de cortesía o fuera de dominio, responden en el exacto idioma de la consulta (`Espejo Lingüístico`), sin invocar el motor de búsqueda vectorial y reduciendo el tiempo de respuesta a menos de 1.5 segundos.
* **Nodo `rewrite_query` (Reescritura de Anáforas):** Si la consulta entra en `LEGAL_RAG`, analiza los turnos previos del historial para reformular la pregunta actual en una frase autocontenida, independiente y sin elipsis.
* **Nodo `retrieve` (Búsqueda Híbrida MMR):** Invoca el índice local `ChromaDB` aplicando **Maximal Marginal Relevance (MMR)** (`top_k=4`, `fetch_k=20`, `lambda_mult=0.7`). Este algoritmo selecciona los fragmentos con mayor relevancia semántica pero maximizando la diversidad entre ellos para evitar la saturación de contexto por párrafos repetidos.
* **Nodo `generate` (Inferencia RAG):** Ensambla el dictamen jurídico aplicando el prompt maestro bilingüe con `gemini-3.1-flash-lite` (`temperature=0.2`).
* **Checkpointer `MemorySaver`:** Persiste el estado y el historial de cada conversación mediante un identificador de hilo (`thread_id`) independiente por usuario o sesión.

---

## Despliegue en Producción (Streamlit Cloud)

El asistente se encuentra desacoplado y en producción como una aplicación web concurrente en Streamlit Cloud:

* **Enlace Público Oficial:** [https://agente-rag-ia-normativa.streamlit.app/](https://agente-rag-ia-normativa.streamlit.app/)
* **Motor Modular (`src/rag_engine.py`):** Encapsula la conexión con ChromaDB, la instanciación de Gemini y la compilación del grafo condicional. Integra parches de compatibilidad automáticos para servidores Linux en la nube (`pysqlite3`).
* **Interfaz Visual (`app.py`):**
  * **Trazabilidad en Vivo (`st.status`):** Muestra en un panel expansible el recorrido técnico del grafo en cada consulta (clasificación del enrutador, reformulación, búsqueda MMR e inferencia del dictamen con su tiempo en segundos).
  * **Aislamiento por Sesión (`uuid.uuid4`):** Cada usuario que abre la aplicación web o reinicia el chat recibe un identificador único que se inyecta en la invocación (`config={"configurable": {"thread_id": st.session_state.thread_id}}`), garantizando la estricta separación de historiales.
  * **Control en Sidebar:** Monitorización de fragmentos indexados, parámetros arquitectónicos y verificación de guardrails activos (sin etiquetas de versionados intermedios visibles en la interfaz).

---

## Suite de Evaluación y Guiones de Demostración en Vivo

El cuaderno ejecuta un banco de 7 pruebas experimentales que validan la solidez arquitectónica en las tres rutas:
1. **Prueba 1 (Consulta Directa RGPD):** Licitud del tratamiento y condiciones del consentimiento (`LEGAL_RAG`).
2. **Prueba 2 (Razonamiento Multi-turno y Anáfora):** Sanciones aplicables al caso anterior sin mencionar explícitamente el RGPD (`LEGAL_RAG` -> `rewrite_query`).
3. **Prueba 3 (IA Agéntica y Chain of Thought):** Auditoría técnica y riesgos de privacidad en bucles de agentes autónomos (`LEGAL_RAG`).
4. **Prueba 4 (Consulta en Inglés sobre EDPB):** Verificación del Espejo Lingüístico respondiendo un dictamen 100% en inglés con traducción del corpus (`LEGAL_RAG`).
5. **Prueba 5 (Guardrail Fuera de Dominio):** Rechazo de baja latencia ante una consulta gastronómica (`OUT_OF_DOMAIN`).
6. **Prueba 6 (Guardrail de Ausencia de Información):** Consulta sobre drones agrícolas en Australia, reconociendo los límites del índice local (`LEGAL_RAG`).
7. **Prueba 7 (Enrutador de Cortesía / Saludo):** Interceptación de un saludo respondiendo de forma natural sin búsqueda (`GENERAL_LLM`).

### Guión de Preguntas para la Demostración en Vivo (`.txt`)
Para acompañar la defensa y demostración en vivo de la aplicación en Streamlit, el repositorio incluye un archivo con consultas exclusivas preparadas para copiar y pegar:
* `preguntas_demo_streamlit_nuevas.txt`: Contiene **5 preguntas nuevas y exclusivas** que incluyen consultas sobre contratos de Encargado de tratamiento (Art. 28 RGPD), desvío por saludo conceptual (microservicios), guardrail ajeno (tarta de queso), búsqueda en inglés sobre responsabilidad en IA (EDPB 2024/28), y una **prueba de memoria multi-turno no consecutiva** donde el nodo reescritor resuelve la elipsis tras un turno intermedio general.

---

## Instrucciones de Ejecución Local

### Prerrequisitos
* **Python:** 3.10 o superior.
* **Clave API de Google Gemini (`GEMINI_API_KEY`):** Credencial activa de Google AI Studio.

### Instalación de Dependencias
En la terminal de comandos situada en la raíz del repositorio:
```bash
pip install -r requirements.txt
```

### Configuración del Entorno (`.env`)
Crea un archivo `.env` en la raíz con tu clave secreta:
```env
GEMINI_API_KEY="AIzaSyTu_Clave_Secreta_De_Google_Gemini_Aqui"
```

### Ejecución de la Interfaz Web (Streamlit)
Lanza la aplicación web local ejecutando:
```bash
streamlit run app.py
```

### Ejecución del Cuaderno Jupyter
1. Abre tu terminal e inicia Jupyter:
   ```bash
   jupyter notebook
   ```
2. Abre `notebooks/proyecto_final_agente_rag._5.ipynb`.
3. Ejecuta las celdas en orden secuencial para verificar la carga del índice `vector_db/`, compilar el grafo LangGraph, generar la imagen del diagrama arquitectónico y consultar las 7 pruebas de evaluación.

---

## Estructura del Repositorio

```text
├── app.py                                 # Interfaz Web interactiva en Streamlit (Despliegue en Cloud)
├── data/
│   └── raw/
│       ├── EDPB_Opinion_2024_28.md        # Dictamen EDPB 2024/28 sobre Modelos de IA (Inglés)
│       ├── Orientaciones Ia Agéntica.md   # Guía Técnica de IA Agéntica y Privacidad (Español)
│       └── reglamentoRGPD.md            # Texto normativo estructurado del RGPD (Español)
├── notebooks/
│   ├── proyecto_final_agente_rag._5.ipynb # Cuaderno Jupyter V5 (Evaluación, Grafo y Chat Interactivo)
│   └── proyecto_final_agente_rag._5.html  # Exportación HTML principal
├── preguntas_demo_streamlit_nuevas.txt    # Batería de 5 preguntas exclusivas para la presentación de Streamlit
├── src/
│   ├── __init__.py                        # Paquete modular
│   └── rag_engine.py                      # Motor RAG LangGraph modular (Router + MMR + MemorySaver)
├── vector_db/
│   └── chroma.sqlite3                     # Base vectorial local pre-indexada (ChromaDB)
├── README.md                              # Documentación oficial del repositorio
└── requirements.txt                       # Dependencias unificadas para producción y desarrollo
```

---

**Créditos y Evaluación:** Proyecto industrializado para la evaluación final del módulo de Inteligencia Artificial Generativa y Agentes RAG dentro del Máster de Formación Permanente en Inteligencia Artificial y Ciencia de Datos. Todos los derechos reservados.
