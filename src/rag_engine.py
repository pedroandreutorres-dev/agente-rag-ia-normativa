import os
import sys
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# Compatibilidad con servidores Linux en Streamlit Cloud (Reemplazo de sqlite3 antiguo por pysqlite3 si está disponible)
if sys.platform.startswith("linux"):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# Cargar variables de entorno locales (.env) o de Streamlit Cloud (st.secrets)
load_dotenv()

def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            pass
    if not key:
        raise ValueError("No se ha encontrado la clave GEMINI_API_KEY ni GOOGLE_API_KEY en el entorno (.env o st.secrets).")
    return key

# Definicion del estado para el grafo de LangGraph (Idéntico a _3.ipynb)
class GraphState(TypedDict):
    messages: List[BaseMessage]
    query_reformulada: str
    documentos_recuperados: List[Any]

# 1. Conexion con ChromaDB usando la colección exacta "corpus_normativo_v3" y modelo "models/gemini-embedding-001"
def get_vectorstore() -> Chroma:
    persist_dir = "vector_db"
    api_key = get_api_key()
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vectorstore = Chroma(
        collection_name="corpus_normativo_v3",
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

# 2. Configuracion exacta de Gemini 3.1 con temperatura 0.2
def get_llm():
    api_key = get_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        max_tokens=1500,
        max_retries=6,
        google_api_key=api_key
    )

# Prompt Maestro exacto al del cuaderno _3.ipynb (Con Espejo Lingüístico y Desduplicación de Citas)
template_rag = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""Eres un Consultor Legal y Tecnológico de Alto Nivel especializado en el Reglamento General de Protección de Datos (RGPD) y en la Gobernanza y Auditoría de Sistemas de Inteligencia Artificial Agéntica.
Tu lenguaje debe ser formal, preciso, analítico, riguroso y estrictamente acotado al conocimiento normativo facilitado.

[REGLAS DE COMPORTAMIENTO ESTRATEGICO Y GUARDRAILS]
1. Responde UNICAMENTE utilizando la información técnica y normativa presente en los fragmentos de [CONTEXTO DE OPERACION RECUPERADO]. No inventes ni supongas nada fuera de estos textos.
2. Si el usuario hace referencia a una pregunta o respuesta anterior, consulta el [HISTORIAL DE CONVERSACION RECIENTE] para mantener coherencia absoluta.
3. GUARDRAIL DE DOMINIO: Si la pregunta del usuario NO está relacionada en absoluto con el ámbito legal, tecnológico, inteligencia artificial o protección de datos (ej. recetas de cocina, deportes, literatura general), debes responder exactamente: "No estoy entrenado para responder sobre ese tema".
4. GUARDRAIL DE AUSENCIA DE INFORMACION: Si la pregunta está dentro del dominio jurídico/tecnológico, pero la respuesta no se encuentra en los fragmentos recuperados, tu única respuesta debe ser: "La información disponible en la base de conocimientos no permite responder a esta consulta".
5. GUARDRAIL DE IDIOMA (ESPEJO LINGÜÍSTICO OBLIGATORIO): Detecta el idioma en que está escrita la "Pregunta actual del usuario" y redacta tu dictamen SIEMPRE en ese exacto mismo idioma:
   - Si la "Pregunta actual del usuario" está en ESPAÑOL, el 100% de tu respuesta (Conclusión Directa, Análisis Normativo y Trazabilidad Jurídica) DEBE estar estrictamente en ESPAÑOL. Queda terminantemente prohibido usar inglés.
   - Si la "Pregunta actual del usuario" está en INGLÉS, el 100% de tu dictamen DEBE redactarse estrictamente en INGLÉS, traduciendo al inglés cualquier concepto legal de los textos en español.

[FORMATO DE SALIDA OBLIGATORIO]
Estructura tu respuesta estrictamente utilizando encabezados Markdown reales SIN NUMERAR y los siguientes tres bloques diferenciados:

### Conclusión Directa
Una única frase corta, profesional y categórica respondiendo de forma clara e inequívoca a la consulta planteada.

### Análisis Normativo
Explicación técnica detallada y fundamentada en viñetas ordenadas o párrafos breves, basada exclusivamente en los preceptos y considerandos de los fragmentos recuperados.

### Trazabilidad Jurídica
Indica al final de tu respuesta las fuentes exactas de donde extrajiste cada afirmación. REGLA DE DESDUPLICACION OBLIGATORIA: Si varios fragmentos utilizados pertenecen exactamente a la misma ley, capítulo y artículo o sección, agrupa las referencias y escribe esa cabecera UNA SOLA VEZ en la lista final. Está estrictamente prohibido repetir citas legales idénticas. Utiliza viñetas tipográficas limpias con el formato:
* **<Nombre_Archivo> (<Titulo_Ley> -> <Capitulo> -> <Articulo>)** — Breve mención al precepto.

[HISTORIAL DE CONVERSACION RECIENTE]
{chat_history}

[CONTEXTO DE OPERACION RECUPERADO]
{context}

Pregunta actual del usuario: 
{question}
"""
)

# Construcción exacta de los 3 Nodos LangGraph idénticos a _3.ipynb
def build_rag_graph(vectorstore: Chroma, llm: ChatGoogleGenerativeAI):
    
    def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
        messages = state["messages"]
        ultima_pregunta = messages[-1].content if messages else ""
        
        if len(messages) <= 1:
            return {"query_reformulada": ultima_pregunta}
        
        prompt_reformulacion = f"""Historial reciente del chat:
{[m.content for m in messages[-4:-1]]}

Pregunta actual del usuario: "{ultima_pregunta}"

Tu tarea tecnica es reformular la pregunta actual para que sea independiente y autocontenida, reemplazando pronombres o referencias implicitas por los conceptos de los turnos anteriores. Devuelve unicamente la pregunta reformulada sin preambulos ni explicaciones."""

        respuesta = llm.invoke([HumanMessage(content=prompt_reformulacion)])
        contenido_req = respuesta.content
        if isinstance(contenido_req, list):
            contenido_req = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido_req])
        return {"query_reformulada": contenido_req.strip()}

    def retrieve_node(state: GraphState) -> Dict[str, Any]:
        query = state.get("query_reformulada") or state["messages"][-1].content
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            return {"documentos_recuperados": docs}
        except Exception as e:
            print(f"[ADVERTENCIA] Error en la consulta de busqueda vectorial: {e}")
            return {"documentos_recuperados": []}

    def generate_node(state: GraphState) -> Dict[str, Any]:
        messages = state["messages"]
        docs = state.get("documentos_recuperados", [])
        query = state.get("query_reformulada") or messages[-1].content
        
        if not docs:
            msg_fallo = "La información disponible en la base de conocimientos no permite responder a esta consulta."
            return {"messages": messages + [AIMessage(content=msg_fallo)]}
            
        contexto_estratificado = []
        for d in docs:
            meta = d.metadata
            src = meta.get("source", "Documento Desconocido")
            titulo_ley = meta.get("titulo_ley", "General")
            cap = meta.get("capitulo", "Sin Capitulo")
            art = meta.get("articulo_o_seccion", "Sin Articulo")
            jerarquia = f"{titulo_ley} -> {cap} -> {art}"
            
            contexto_estratificado.append(f"--- DOCUMENTO: {src} | ESTRUCTURA: [{jerarquia}] ---\n{d.page_content}")
            
        texto_contexto = "\n\n".join(contexto_estratificado)
        
        historial_str = ""
        if len(messages) > 1:
            turnos_recientes = messages[-5:-1] if len(messages) > 5 else messages[:-1]
            for m in turnos_recientes:
                rol = "Usuario" if isinstance(m, HumanMessage) else "Consultor"
                historial_str += f"{rol}: {m.content}\n"
                
        prompt_evaluado = template_rag.format(
            chat_history=historial_str or "Sin historial previo.",
            context=texto_contexto,
            question=query
        )
        
        respuesta = llm.invoke([HumanMessage(content=prompt_evaluado)])
        contenido = respuesta.content
        if isinstance(contenido, list):
            contenido = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido])
        return {"messages": messages + [AIMessage(content=contenido)]}

    workflow = StateGraph(GraphState)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
