from typing import Any, Dict, List

import chromadb
from rag_chain import VectorDBInterface

class ChromaAdapter(VectorDBInterface):
    """Adapter for the ChromaDB client to provide a unified interface for vector databases."""
    def __init__(self, path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)

    def create_collection(self, collection_name: str, dimension: int = 0):
        self.client.get_or_create_collection(name=collection_name)

    def upsert(self, collection_name: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        collection = self.client.get_collection(name=collection_name)
        collection.upsert(ids=ids, embeddings=vectors, metadatas=payloads)

    def search(self, collection_name: str, query_vector: List[float], filters: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        collection = self.client.get_collection(name=collection_name)
        metadata_filters = self.prepare_filters(filters) 
        results = collection.query(query_embeddings=[query_vector], where=metadata_filters, n_results=top_k)

        normalized = []
        if results["ids"]:
            for idx, doc_id in enumerate(results['ids'][0]):
                normalized.append({
                    "id": doc_id,
                    "payload": results['metadatas'][0][idx] if results['metadatas'] else {},
                    "score": results['distances'][0][idx] if results['distances'] else None
                })

        return normalized

    def prepare_filters(self, filters: Dict) -> Dict:
        """Format the filter settings passed from the frontend."""
        conditions = []

        # Date range modified
        modified_range = filters.get("modified_date_range")
        if modified_range and len(modified_range) == 2:
            if modified_range[0] is not None:
                conditions.append({"modified": {"$gte": modified_range[0]}})
            if modified_range[1] is not None:
                conditions.append({"modified": {"$lte": modified_range[1]}})

        # Date range created
        created_range = filters.get("created_date_range")
        if created_range and len(created_range) == 2:
            if created_range[0] is not None:
                conditions.append({"created": {"$gte": created_range[0]}})
            if created_range[1] is not None:
                conditions.append({"created": {"$lte": created_range[1]}})

        # Tags
        tags = filters.get("selected_tags")
        if tags:
            tag_conditions = [{"tags": {"$contains": tag for tag in filters["selected_tags"]}}]

            if len(tag_conditions) == 1:
                conditions.append(tag_conditions[0])
            else:
                conditions.append({"$or": tag_conditions})

        # Document Types
        doc_types = filters.get("selected_document_types")
        if doc_types:
            conditions.append({"document_type": {"$in": filters["selected_document_types"]}})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
