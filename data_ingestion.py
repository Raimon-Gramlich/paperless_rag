import os
from typing import Dict
import json
from datetime import datetime, timedelta, timezone
import urllib3
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import bm25s
import Stemmer
from paperless_api import fetch_data
from vectordb_adapters.chroma_adapter import ChromaAdapter
from vectordb_adapters.qdrant_adapter import QdrantAdapter
from rag_chain import CHROMA_DB_STRING, QDRANT_DB_STRING
import uuid

# Suppress only the InsecureRequestWarning from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

NAMESPACE_MY_APP = uuid.NAMESPACE_DNS

DOC_LANG = "German"

if DOC_LANG == "German":
    STEMMER_LANG = "german"
    BM25S_LANG = "de"
else:
    STEMMER_LANG = "english"
    BM25S_LANG = "en"

PAPERLESS_API_URL = os.getenv("PAPERLESS_API_URL")
LOCAL_VLLM_BASE_URL = os.getenv("LOCAL_VLLM_BASE_URL")
LOCAL_VLLM_API_KEY = os.getenv("LOCAL_VLLM_API_KEY", "")
LOCAL_EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "missing_embedding_model_name")

# Configuration
DB_PATH = "./chroma_db"
SYNC_STATE_FILE = "sync_state.json"
BM25_INDEX_PATH = "./bm25_index"

VECTORDB_NAME = "paperless_vectordb"

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model=LOCAL_EMBEDDING_MODEL_NAME,
    base_url=LOCAL_VLLM_BASE_URL,
    api_key=LOCAL_VLLM_API_KEY
)

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

stemmer = Stemmer.Stemmer(STEMMER_LANG)

def get_sync_state() -> Dict:
    """Load the synced document IDs with their respective last ingested timestamps from a local file."""
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_sync_state(sync_state: dict):
    """Save the last synced document ID and timestamp to a local file."""
    with open(SYNC_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sync_state, f)

def ingest_data(vector_db_string, qdrant_url=None):
    """Main ingestion pipeline with Sync logic."""
    data = fetch_data()
    if not data:
        print("No documents found.")
        return
    try:
        documents = data["results"]
    except KeyError as e:
        raise KeyError("Expected dict with key 'results' for the list of documents.") from e

    sync_state = get_sync_state()

    # Sort documents by ID to ensure we process them in order
    documents.sort(key=lambda x: x.get("id", 0))

    new_docs_count = 0
    updated_docs_count = 0

    all_chunks = []

    for doc in documents:
        doc_id = doc.get("id", 0)
        doc_updated = datetime.fromisoformat(doc.get("modified", "1970-01-01T00:00:01.000000+02:00"))
        last_ingested = datetime.fromisoformat(sync_state.get(doc_id, "1970-01-01T00:00:00.000000+02:00"))

        # Sync logic: Only process if it's new or updated since last sync
        # Check if the updated timestamp is greater than last_updated.
        # Create pairs of id and last modified timestamps. Example: 2026-08-06T21:09:22.636620+02:00
        if doc_updated > last_ingested:
            print(f"Processing document {doc_id}: {doc.get('title')}")
            content = doc["content"]

            if content:
                print(f"--- Content length for doc {doc_id}: {len(content)} chars ---")
                chunks = text_splitter.split_text(content)
                if chunks:
                    print(f"--- First chunk snippet for doc {doc_id}: ---")
                    print(chunks[0][:500])
                    print("-" * 40)
                for part_id, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source_id": doc_id,
                            "part_id": part_id,
                            "title": doc.get("title", "Untitled"),
                            "url": f"{PAPERLESS_API_URL}/documents/{doc_id}/",
                            "created": int(datetime.fromisoformat(doc.get("created", "1970-01-01T00:00:01.000000+02:00")).timestamp()),
                            "modified": int(datetime.fromisoformat(doc.get("modified", "1970-01-01T00:00:01.000000+02:00")).timestamp()),
                            "document_type": doc.get("document_type", "no_document_type"),
                            "tags": doc.get("tags", [])
                        }
                    })

                if doc_id in sync_state.keys():
                    updated_docs_count += 1
                else:
                    new_docs_count += 1

                # Update local sync state
                sync_state[doc_id] = datetime.now(timezone(timedelta(hours=2))).isoformat()

    if not all_chunks:
        print("No new or updated content to ingest.")
        return

    print(f"Total new chunks created: {len(all_chunks)}")

    # Store in Vector DB
    print(f"Storing chunks in {vector_db_string}...")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    query_vectors = embeddings.embed_documents(texts)
    ids = [str(uuid.uuid5(NAMESPACE_MY_APP, f"doc_{m['source_id']}_part_{m['part_id']}")) for m in metadatas]
    
    # Build and save BM25 index
    print(f"Building BM25 index at {BM25_INDEX_PATH}...")
    corpus_tokens = bm25s.tokenize(texts, stopwords=BM25S_LANG, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(BM25_INDEX_PATH, corpus=all_chunks)

    print(f"====== {vector_db_string} =====")

    if vector_db_string == CHROMA_DB_STRING:
        vector_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=VECTORDB_NAME,
            persist_directory=DB_PATH
        )
    elif vector_db_string == QDRANT_DB_STRING:
        if qdrant_url:
            vector_adapter = QdrantAdapter(url=qdrant_url)
            vector_adapter.create_collection(VECTORDB_NAME, 1024)
            vector_adapter.upsert(
                collection_name=VECTORDB_NAME,
                ids=ids,
                vectors=query_vectors,
                payloads=[{
                    "page_content": chunk["text"],
                    "metadata": chunk["metadata"],
                } for chunk in all_chunks ],
            )
        else:
            raise TypeError("No Qdrant server url given even though Qdrant is selected.")

    else:
        raise ValueError(f"Unkown vector database name: {vector_db_string}")


    save_sync_state(sync_state)
    print(f"Ingestion complete. Processed {new_docs_count} new and {updated_docs_count} updated documents.")

if __name__ == "__main__":
    ingest_data("ChromaDB")
