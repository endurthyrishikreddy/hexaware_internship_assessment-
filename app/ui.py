# --------------------------------------------------------------------------
# File: app/ui.py
# Description: A simple Streamlit web interface for the RAG system.
# --------------------------------------------------------------------------
import streamlit as st
import requests

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/query"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="RAG Internship Project", layout="wide")
st.title("📄 RAG System with Elasticsearch and Ollama")

# --- UI Components ---
with st.sidebar:
    st.header("Configuration")
    retrieval_mode = st.radio(
        "Select Retrieval Mode:",
        ("Hybrid (ELSER + Dense + BM25)", "ELSER-only"),
        index=0
    )
    
    st.info(
        "**Hybrid Mode:** Combines keyword, semantic, and sparse vector search for the best relevance.\n\n"
        "**ELSER-only Mode:** Uses Elastic's sparse vector model for semantic search."
    )

# Map UI selection to API parameter
mode_map = {
    "Hybrid (ELSER + Dense + BM25)": "hybrid",
    "ELSER-only": "elser"
}
selected_mode = mode_map[retrieval_mode]

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # --- Call FastAPI Backend ---
                payload = {"question": prompt, "mode": selected_mode}
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status() # Raise an exception for bad status codes
                
                result = response.json()
                answer = result.get("answer", "Sorry, something went wrong.")
                sources = result.get("sources", [])
                
                # --- Display Answer and Sources ---
                message_placeholder.markdown(answer)
                
                if sources:
                    st.subheader("Sources:")
                    for source in sources:
                        with st.expander(f"Source: {source['metadata'].get('filename', 'Unknown')}"):
                            st.text(f"Relevance Score: {source['metadata'].get('_score', 'N/A')}")
                            st.markdown(source['page_content'])
                            
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API. Is it running? Error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
        full_response = answer
        st.session_state.messages.append({"role": "assistant", "content": full_response})
