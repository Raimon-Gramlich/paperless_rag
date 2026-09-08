import streamlit as st
from rag_chain import CHROMA_DB_STRING, QDRANT_DB_STRING

def setup_page():
    """Configure the Streamlit page layout."""
    st.set_page_config(page_title="Settings", page_icon="⚙️")
    st.title("📄 Paperless-ngx Document Chat")
    st.markdown("Chat with your personal documents stored in Paperless-ngx.")

def settings_page():
    """Defines the main content of the settings page."""
    st.markdown("# Settings")
    st.markdown("## OpenAI-compatible API Config")
    st.markdown("### Chatbot / Agent LLM")
    st.text_input("LLM URL", key="llm_api_url")
    st.text_input("LLM API Key", key="llm_api_key",type="password")
    st.text_input("LLM Model Name", key="llm_model_name")

    st.markdown("### Embedding Model")
    st.text_input("Embedding Model URL", key="embedding_api_url")
    st.text_input("Embedding Model API Key", key="embedding_api_key",type="password")
    st.text_input("Embedding Model Name", key="embedding_model_name")

    st.markdown("### Renraking Model")
    st.toggle("Enable Reranking", True, key="reranking_enabled")
    st.text_input(
        "Reranking Model URL",
        key="reranking_api_url",
        disabled=not st.session_state.reranking_enabled
    )
    st.text_input(
        "Reranking Model API Key",
        key="reranking_api_key",
        type="password",
        disabled=not st.session_state.reranking_enabled
    )
    st.text_input(
        "Reranking Model Name",
        key="reranking_model_name",
        disabled=not st.session_state.reranking_enabled
    )

    st.markdown("## Vector Database Config")
    st.radio(
        f"Using {st.session_state.vectordb_string} for vector storage.",
        [CHROMA_DB_STRING, QDRANT_DB_STRING],
        key="vectordb_string"
    )

    st.text_input(
        "Qdrant Server URL",
        key="qdrant_api_url",
        disabled=st.session_state.vectordb_string != QDRANT_DB_STRING
    )

    st.text_input(
        "Qdrant API Key",
        key="qdrant_api_key",
        type="password",
        disabled=st.session_state.vectordb_string != QDRANT_DB_STRING
    )

#setup_page()
settings_page()
