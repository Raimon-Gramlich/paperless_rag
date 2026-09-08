import logging
import streamlit as st
from data_ingestion import ingest_data
from paperless_api import fetch_document_types, fetch_tags
from rag_chain import CHROMA_DB_STRING

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_page():
    """Configure the Streamlit page layout."""
    st.set_page_config(page_title="Paperless-ngx RAG Chat", page_icon="📄")
    st.title("📄 Paperless-ngx Document Chat")
    st.markdown("Chat with your personal documents stored in Paperless-ngx.")

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tags" not in st.session_state:
        st.session_state.tags = fetch_tags()
    if "doc_types" not in st.session_state:
        st.session_state.doc_types = fetch_document_types()
    if "selected_tags" not in st.session_state:
        st.session_state.selected_tags = []
    if "selected_doc_types" not in st.session_state:
        st.session_state.selected_doc_types = []
    if "selected_modified_date_range" not in st.session_state:
        st.session_state.selected_modified_date_range = []
    if "selected_created_date_range" not in st.session_state:
        st.session_state.selected_created_date_range = []
    if "reranking_enabled" not in st.session_state:
        st.session_state.reranking_enabled = False
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_api_url" not in st.session_state:
        st.session_state.llm_api_url = ""
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = ""
    if "embedding_api_key" not in st.session_state:
        st.session_state.embedding_api_key = ""
    if "embedding_api_url" not in st.session_state:
        st.session_state.embedding_api_url = ""
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = ""
    if "reranking_enabled" not in st.session_state:
        st.session_state.reranking_enabled = False
    if "reranking_api_key" not in st.session_state:
        st.session_state.reranking_api_key = ""
    if "reranking_api_url" not in st.session_state:
        st.session_state.reranking_api_url = ""
    if "reranking_model_name" not in st.session_state:
        st.session_state.reranking_model_name = ""
    if "vectordb_string" not in st.session_state:
        st.session_state.vectordb_string = CHROMA_DB_STRING
    if "qdrant_api_url" not in st.session_state:
        st.session_state.qdrant_api_url = ""
    if "qdrant_api_key" not in st.session_state:
        st.session_state.qdrant_api_key = ""

def render_sidebar():
    """Render the sidebar controls."""
    with st.sidebar:
        st.header("Metadata Filters")
        st.pills("Document Types", st.session_state.doc_types.keys(), selection_mode="multi",
                format_func=lambda key: st.session_state.doc_types[key], key="selected_doc_types")
        st.pills("Tags", st.session_state.tags.keys(), selection_mode="multi",
                format_func=lambda key: st.session_state.tags[key], key="selected_tags")
        st.markdown("### Select a date range to filter for:")
        st.date_input(
            "Created",
            value=[],
            max_value="today",
            key="selected_created_date_range"
            )
        st.date_input(
            "Last modified",
            value=[],
            max_value="today",
            key="selected_modified_date_range"
            )
        st.header("Controls")
        if st.button("Sync Now"):
            with st.spinner("Syncing documents..."):
                try:
                    ingest_data(st.session_state.vectordb_string, st.session_state.qdrant_api_url)
                    st.success("Sync complete!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")
                    logger.error("Sync error: %s", e)




def main():
    """Main entry point for the Streamlit application."""
    setup_page()
    initialize_session_state()

    # Define pages
    main_page = st.Page("views/1_Chat.py", title="Home", icon="💬", default=True)
    settings_page = st.Page("views/2_Settings.py", title="Settings", icon="⚙️")

    # Setup navigation
    pg = st.navigation([main_page, settings_page])

    # Shared sidebar elements that stay across pages
    st.sidebar.title("Global Controls")
    render_sidebar()

    pg.run()

if __name__ == "__main__":
    main()
