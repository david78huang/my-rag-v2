from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import QDRANT_HOST, QDRANT_PORT

class QdrantStore:

    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )

    def upsert(self, collection, point_id, vector, payload):

        self.client.upsert(
            collection_name=collection,
            points=[{
                "id": point_id,
                "vector": vector,
                "payload": payload
            }]
        )

    def ensure_qdrant_collection(client: QdrantClient, collection: str, dim: int = 1024):
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def search(self, collection, vector, user_id, topk=10):
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

        resp = self.client.query_points(
            collection_name=collection,
            query=vector,
            limit=topk,
            with_payload=True,
            query_filter=query_filter
        )

        return getattr(resp, "points", resp)