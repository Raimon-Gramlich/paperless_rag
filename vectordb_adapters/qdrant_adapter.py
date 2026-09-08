from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Condition, FieldCondition, Filter, VectorParams, PointStruct, Range, MatchAny, SparseVectorParams, Modifier, Prefetch, FusionQuery, Fusion, Document, PayloadSchemaType
from rag_chain import VectorDBInterface

class QdrantAdapter(VectorDBInterface):
    """Adapter for the Qdrant client to provide a unified interface for vector databases."""
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url, check_compatibility=False)

    def create_collection(self, collection_name: str, dimension: int):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=dimension, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF)
                }
            )

            fields_to_index = {
                "metadata[].tags": PayloadSchemaType.KEYWORD,
                "metadata[].document_type": PayloadSchemaType.KEYWORD,
                "metadata[].created": PayloadSchemaType.INTEGER,
                "metadata[].modified": PayloadSchemaType.INTEGER
            }

            for field_name, field_schema in fields_to_index.items():
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

        print("Collection created or already existed.")

    def upsert(self, collection_name: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):

        points = [
                PointStruct(id=idx, vector={"dense": vector, "bm25": Document(text=payload["page_content"], model="Qdrant/bm25")}, payload=payload) for idx, vector, payload in zip(ids, vectors, payloads)
        ]

        print(points[1:5])

        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name: str, query_vector: List[float], filters: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if filters:
            metadata_filters = Filter(
                must=self.prepare_filters(filters)
            )
        else:
            metadata_filters = None

        prefetch_list = [
            Prefetch(
                query=query_vector,
                using="dense",
                limit=top_k
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=top_k
            )
        ]

        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_list,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=metadata_filters,
            limit=top_k
        ).points

        return results

    def prepare_filters(self, filters: Dict) -> List[Condition]:
        """Format the filter settings passed from the frontend."""
        conditions = []

        # Date range modified
        modified_range = filters.get("modified_date_range")
        if modified_range and len(modified_range) == 2:
            modified_range_args = {}
            if modified_range[0] is not None:
                modified_range_args["gte"] = modified_range[0]
            if modified_range[1] is not None:
                modified_range_args["lte"] = modified_range[1]

            conditions.append(FieldCondition(key="modified", range=Range(**modified_range_args)))

        # Date range created
        created_range = filters.get("created_date_range")
        if created_range and len(created_range) == 2:
            created_range_args = {}
            if created_range[0] is not None:
                created_range_args["gte"] = created_range[0]
            if created_range[1] is not None:
                created_range_args["lte"] = created_range[1]
            conditions.append(FieldCondition(key="created", range=Range(**created_range_args)))

        # Tags
        tags = filters.get("selected_tags")
        if tags:
            conditions.append(FieldCondition(key="tags", match=MatchAny(any=filters["selected_tags"])))

        # Document Types
        doc_types = filters.get("selected_document_types")
        if doc_types:
            conditions.append(FieldCondition(key="document_type", match=MatchAny(any=filters["selected_document_types"])))

        return conditions
