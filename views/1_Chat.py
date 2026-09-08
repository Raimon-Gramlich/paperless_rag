import logging
import streamlit as st
from rag_chain import ask_question

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_chat_history():
    """Display the chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_input():
    """Handle user input and generate assistant responses."""
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                metadata_filters = {
                    "modified_date_range": st.session_state.selected_modified_date_range,
                    "created_date_range": st.session_state.selected_created_date_range,
                    "selected_tags": [tag for tag in st.session_state.selected_tags],
                    "selected_document_types": [doc_type for doc_type in st.session_state.selected_doc_types]
                }
                try:
                    response = ask_question(
                        prompt,
                        metadata_filters,
                        st.session_state.reranking_enabled,
                        st.session_state.vectordb_string,
                        url=st.session_state.qdrant_api_url
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error("Error during ask_question: %s", e, exc_info=True)

display_chat_history()
handle_chat_input()
