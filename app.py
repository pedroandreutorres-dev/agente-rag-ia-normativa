import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_engine import get_vectorstore, get_llm, build_rag_graph

# 1. Configuracion de pagina en Streamlit
st.set_page_config(
    page_title="Sistema RAG Conversacional - RGPD / IA Agéntica",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz moderna y profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .status-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Inicializacion en Cache del Motor RAG
@st.cache_resource(show_spinner="Conectando con base vectorial local e instanciando motor RAG LangGraph...")
def init_system():
    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        graph = build_rag_graph(vectorstore, llm)
        num_docs = len(vectorstore.get()["ids"])
        return graph, num_docs, None
    except Exception as e:
        return None, 0, str(e)

graph_app, num_fragmentos, err_init = init_system()

# 3. Barra Lateral (Sidebar) de Gobernanza y Control del Sistema
with st.sidebar:
    st.markdown("### Estado del Sistema")
    if err_init:
        st.error(f"Error al inicializar el índice: {err_init}")
    else:
        st.success("Conectado y listo en memoria")
        st.metric(label="Fragmentos Indexados (ChromaDB)", value=f"{num_fragmentos} trozos")
        
    st.markdown("---")
    st.markdown("### Parámetros Arquitectónicos")
    st.markdown("- **Motor de Inferencia:** `gemini-3.1-flash-lite`")
    st.markdown("- **Embeddings:** `models/gemini-embedding-001`")
    st.markdown("- **Colección:** `corpus_normativo_v3`")
    st.markdown("- **Temperatura LLM:** `0.2` (Alta fidelidad)")
    st.markdown("- **Búsqueda Vectorial:** `top_k = 4`")
    st.markdown("- **Orquestador:** `LangGraph (StateGraph)`")
    
    st.markdown("---")
    st.markdown("### Guardrails de Seguridad Activos")
    st.markdown("1. **Dominio Normativo:** Rechazo exterior al ámbito legal/tecnológico.")
    st.markdown("2. **Ausencia de Información:** Rechazo ante lagunas en el corpus local.")
    st.markdown("3. **Espejo Lingüístico:** Respuesta estricta en el idioma de la consulta.")
    st.markdown("4. **Desduplicación de Citas:** Agrupación legal por artículo y sección.")
    st.markdown("5. **Memoria Deslizante:** Ventana de retención de últimos 4 turnos.")
    
    st.markdown("---")
    if st.button("Reiniciar Historial de Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Cabecera Principal
st.markdown('<div class="main-header">Sistema RAG Conversacional sobre Regulación e Inteligencia Artificial Agéntica</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asistente jurídico y técnico orquestado con LangGraph, consultando en tiempo real el RGPD y los dictámenes oficiales del EDPB.</div>', unsafe_allow_html=True)

if err_init:
    st.stop()

# 5. Gestión del Historial en Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderizar mensajes guardados en el historial
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 6. Interacción en Vivo con el Usuario
prompt = st.chat_input("Escribe tu consulta normativa o técnica aquí...")

if prompt:
    # Mostrar pregunta del usuario al instante
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Orquestar grafo de LangGraph visualizando la trazabilidad en un st.status
    with st.chat_message("assistant"):
        with st.status("Orquestando grafo LangGraph y consultando base vectorial ChromaDB...", expanded=True) as status:
            t0 = time.time()
            st.write("1. **Nodo `rewrite_query`:** Reformulando anáforas y elipsis desde el historial reciente...")
            
            st.write("2. **Nodo `retrieve`:** Consultando el índice local ChromaDB para extraer los 4 fragmentos jurídicos más relevantes...")
            
            st.write("3. **Nodo `generate`:** Evaluando guardrails normativos, aplicando espejo lingüístico e infiriendo dictamen con `gemini-3.1-flash-lite`...")
            
            estado_input = {"messages": st.session_state.messages}
            try:
                estado_salida = graph_app.invoke(estado_input)
                t_tot = time.time() - t0
                status.update(label=f"Trazabilidad técnica del grafo completada en {t_tot:.2f}s", state="complete", expanded=False)
            except Exception as exc:
                t_tot = time.time() - t0
                status.update(label="Advertencia en la consulta al motor RAG", state="error", expanded=True)
                msg_error = f"[ADVERTENCIA] Ha ocurrido un error al procesar la consulta: {exc}\n\n*Sugerencia: Si la API de Google está temporalmente saturada, espera unos segundos e inténtalo de nuevo.*"
                st.error(msg_error)
                st.stop()
        
        # SIEMPRE FUERA Y DEBAJO DEL ST.STATUS: Renderizado principal de la respuesta
        # De esta forma el dictamen SIEMPRE queda visible de inmediato y el acordeón cerrado solo oculta los 3 pasos técnicos
        respuesta_agente = estado_salida["messages"][-1].content
        if isinstance(respuesta_agente, list):
            respuesta_agente = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in respuesta_agente])
            
        st.markdown(respuesta_agente)
        st.session_state.messages.append(AIMessage(content=respuesta_agente))
