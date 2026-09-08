import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
import bm25s
import Stemmer

# Configuration
CHROMA_DB_STRING = "ChromaDB"
QDRANT_DB_STRING = "Qdrant"

DOC_LANG = "German"

if DOC_LANG == "German":
    STEMMER_LANG = "german"
    BM25S_LANG = "de"
else:
    STEMMER_LANG = "english"
    BM25S_LANG = "en"

# Load environment variables
load_dotenv()

LOCAL_VLLM_BASE_URL = os.getenv("LOCAL_VLLM_BASE_URL")
LOCAL_VLLM_API_KEY = os.getenv("LOCAL_VLLM_API_KEY", "")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME")
LOCAL_EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
LOCAL_RERANKING_MODEL_NAME = os.getenv("LOCAL_RERANKING_MODEL_NAME")
LOCAL_RERANKING_URL = os.getenv("LOCAL_RERANKING_URL")

# Configuration
DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "bm25_index"
RERANK = True

stemmer = Stemmer.Stemmer(STEMMER_LANG)

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model=LOCAL_EMBEDDING_MODEL_NAME,
    base_url=LOCAL_VLLM_BASE_URL,
    api_key=LOCAL_VLLM_API_KEY
)

# Initialize LLM
llm = ChatOpenAI(
    model=LOCAL_LLM_MODEL_NAME,
    base_url=LOCAL_VLLM_BASE_URL,
    api_key=LOCAL_VLLM_API_KEY,
    temperature=0.7
)

# Load Vector Store
vector_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

class VectorDBInterface(ABC):
    """Unified interface for interacting with vector databases."""

    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int):
        pass

    @abstractmethod
    def upsert(self, collection_name: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], filters: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def prepare_filters(self, filters: Dict) -> Any:
        pass

class BM25SRetriever(BaseRetriever):
    """A custom LangChain sparse retriever powered by the high-performance bm25s library."""
    model_config = ConfigDict(arbitrary_types_allowed = True)

    documents: List[Document]
    retriever_obj: bm25s.BM25
    k: int = 4
    lang: str = "german"

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 4, method: str = "lucene", **kwargs):
        """Initializes the bm25s index from a list of LangChain Documents."""
        # 1. Extract string contents
        corpus_texts = [doc.page_content for doc in documents]

        # 2. Tokenize text using bm25s built-in lexer
        tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords=cls.lang)

        # 3. Create and fit the index matrix
        retriever_obj = bm25s.BM25(method=method, **kwargs)
        retriever_obj.index(tokenized_corpus)

        return cls(documents=documents, retriever_obj=retriever_obj, k=k)

    @classmethod
    def from_disk(cls, index_path: str = BM25_INDEX_PATH, k: int = 4, **kwargs):
        retriever = bm25s.BM25.load(index_path, load_corpus=True, mmap=True)

        documents = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in retriever.corpus]

        return cls(documents=documents, retriever_obj=retriever, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Executes the sparse keyword retrieval."""
        # Tokenize the user query
        tokenized_query = bm25s.tokenize(query, stopwords=self.lang, stemmer=stemmer)

        # Retrieve the indices of the top matched documents
        relevant_docs, scores = self.retriever_obj.retrieve(tokenized_query, k=self.k, corpus=self.documents)

        return relevant_docs


# Create BM25 retriever
bm25_retriever = None
BM25SRetriever.lang = BM25S_LANG
if os.path.exists(BM25_INDEX_PATH):
    bm25_retriever = BM25SRetriever.from_disk(BM25_INDEX_PATH)
    bm25_retriever.k = 5

# Create retrievers
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Translation logic for cross-lingual RAG (English query to German documents)
translation_prompt = ChatPromptTemplate.from_template(
    "Translate the following English query into German. Provide only the translated text without any additional commentary.\n\nQuery: {question}"
)
translation_chain = translation_prompt | llm | StrOutputParser()

# Define System Prompt
template = """You are a helpful assistant that answers questions based strictly on the provided context from the user's personal documents.

Context:
{context}

Question: {question}

Instructions:
1. Use ONLY the provided context to answer the question.
2. If the answer is not contained within the context, state that you do not have enough information to answer.
3. For every piece of information you provide, cite the source by including the Document ID and the URL at the end of the sentence or paragraph.
4. Be concise and direct.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Define the RAG Chain
def format_docs(docs):
    """Format the retrieved documents for the prompt."""
    formatted = []
    print(docs)
    for doc in docs["documents"]:
        print("loop")
        print(doc)
        content = f"Source ID: {doc.metadata.get('source_id')}, URL: {doc.metadata.get('url')}\nContent: {doc.page_content}"
        formatted.append(content)
    return "\n\n".join(formatted)

def rerank(ctx: dict, top_n: int = 3) -> List[Document]:
    """Reranks retrieved context using cross-encoder reranker."""
    query = ctx["query"]
    documents: list[Document] = ctx["documents"]

    if not documents:
        return []

    # Map LangChain document items into raw text string candidates
    input_texts = [doc.page_content for doc in documents]

    # Structure payload following standard vLLM / Jina API contracts
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LOCAL_VLLM_API_KEY}"
    }
    payload = {
        "model": LOCAL_RERANKING_MODEL_NAME,
        "query": query,
        "documents": input_texts,
        "top_n": top_n
    }

    try:
        response = requests.post(LOCAL_RERANKING_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"data: {data}")

        reranked_docs = []
        for item in data.get("results", []):
            original_idx = item["index"]
            score = item["relevance_score"]

            # Fetch document, apply new score tracking to its metadata
            doc = documents[original_idx]
            doc.metadata["relevance_score"] = score
            reranked_docs.append(doc)

        return reranked_docs

    except Exception as e:
        print(f"Rerank Error: {e}. Falling back to unranked raw pool.")
        return documents[:top_n]

def rrf_merge(scores: dict, k=60, sparse_weight=0.5, dense_weight=0.5) -> List[Document]:
    """Combines rankings of retrievers using Reciprocal Rank Fusion."""
    sparse_docs = scores["sparse"][0]
    dense_docs = scores["dense"]

    rrf_scores = {}
    doc_map = {}

    print("=== SPARSE ===")
    print(sparse_docs)
    print("=== DENSE ===")
    print(dense_docs)

    for source_docs, weight in [(sparse_docs, sparse_weight), (dense_docs, dense_weight)]:
        for rank, doc in enumerate(source_docs):
            metadata = doc.metadata
            doc_id = f"{metadata.get("source_id")}-{metadata.get("part_id")}"
            doc_map[doc_id] = doc

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0

            rrf_scores[doc_id] += weight * (1.0 / (k + (rank + 1)))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    print("=== SORTED DOCS ===")
    print(sorted_docs)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]

def merge_and_prepare_rerank(pipeline_input: dict) -> dict:
    """Extracts previous pipeline step's output and fuses the rankings."""
    raw_query = pipeline_input["query"]
    ranks = pipeline_input["retrieved_results"]

    fused_ranks = rrf_merge(ranks)

    return {
        "documents": fused_ranks,
        "query": raw_query
    }

def dynamic_retrieval(inputs):
    """Constructs the retrieval chain dynamically with custom filters based on the user settings."""
    translated_query = translation_chain.invoke(inputs["question"])
    current_filter = inputs["metadata_filters"]
    reranking_enabled = inputs.get("reranking_enabled", RERANK)
    if bm25_retriever:
        retrieval_stage_2 = (
            RunnableParallel(
                retrieved_results=RunnableParallel(
                    sparse=bm25_retriever,
                    dense=vector_retriever.with_config(
                        search_kwargs={"filter": current_filter}
                    )
                ),
                query=RunnablePassthrough()
            )
            | RunnableLambda(merge_and_prepare_rerank)
        ).invoke(translated_query)
    else:
        retrieval_stage_2 = (
            {
                "documents": vector_retriever.with_config(
                        search_kwargs={"filter": current_filter}
                    ).invoke(translated_query),
                "query": RunnablePassthrough()
            }
        )

    docs = retrieval_stage_2

    print(docs)

    if reranking_enabled:
        docs = rerank(docs)

    return format_docs(docs)

if bm25_retriever:
    retrieval_stage = (
        RunnableParallel(
            retrieved_results=RunnableParallel(
                sparse=bm25_retriever,
                dense=vector_retriever
            ),
            query=RunnablePassthrough()
        )
        | RunnableLambda(merge_and_prepare_rerank)
    )
else:
    retrieval_stage = (
        {"documents": vector_retriever, "query": RunnablePassthrough()}
    )

if RERANK:
    pipeline = retrieval_stage | RunnableLambda(rerank)
else:
    pipeline = retrieval_stage

rag_chain = (
    {
        "context": (
            RunnablePassthrough()
            | translation_chain
            | pipeline
            | format_docs
        ),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

dynamic_rag_chain = (
    {
        "context": RunnableLambda(dynamic_retrieval),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

def dynamic_qdrant(inputs):
    """Constructs the retrieval chain dynamically with custom filters based on the user settings."""
    from vectordb_adapters.qdrant_adapter import QdrantAdapter
    translated_query = translation_chain.invoke(inputs["question"])
    current_filter = inputs["metadata_filters"]
    reranking_enabled = inputs.get("reranking_enabled", RERANK)

    adapter = QdrantAdapter(url="")

    query_vector = embeddings.embed_query(translated_query)
    docs = adapter.search("", filters=current_filter, query_vector=query_vector, query=translated_query)

    docs = {
            "query": translated_query,
            "documents": [Document(page_content=doc.get("paget_content", ""), metadata=doc.get("metadata", "")) for doc in docs]
        }

    if reranking_enabled:
        docs = rerank(docs)

    return format_docs(docs)

dynamic_qdrant_rag_chain = (
    {
        "context": RunnableLambda(dynamic_qdrant),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(question: str, filters: Dict, reranking_enabled: bool, vectordb_string: str, url: str) -> str:
    """Query the RAG system."""
    print(f"Querying: {question}")
    print(f"filters: {filters}")

    if vectordb_string == CHROMA_DB_STRING:
        from vectordb_adapters.chroma_adapter import ChromaAdapter
        vectordb_adapter = ChromaAdapter() 

        metadata_filters = vectordb_adapter.prepare_filters(filters)

        response = dynamic_rag_chain.invoke({
            "question": question,
            "metadata_filters": metadata_filters,
            "reranking_enabled": reranking_enabled
        })
    elif vectordb_string == QDRANT_DB_STRING:
        from vectordb_adapters.qdrant_adapter import QdrantAdapter
        vectordb_adapter = QdrantAdapter(url)

        response = dynamic_qdrant_rag_chain.invoke({
            "question": question,
            "metadata_filters": filters,
            "reranking_enabled": reranking_enabled 
        })
    else:
        raise ValueError(f"Unknown vector database name: {vectordb_string}")

    return response

if __name__ == "__main__":
    # Test query for cross-lingual retrieval
    QUERY = "How much costs a Seagate HDD?"
    print(ask_question(QUERY, {}, True, "ChromaDB", ""))
