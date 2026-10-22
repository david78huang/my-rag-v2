import hashlib
from vector.qdrant import QdrantStore
from vector.embed import Embedder
import logging
logger = logging.getLogger("uvicorn.error")
class SemanticMemory:

    def __init__(self):
        self.qdrant = QdrantStore()
        self.embed = Embedder()

    def add(self, user_id, text):

        vec = self.embed.encode(text)
        logger.info("write vec into qdrant")
        logger.info(text)
        pid = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10**12

        self.qdrant.upsert(
            "memory",
            pid,
            vec,
            {
                "user_id": user_id,
                "text": text
            }
        )

    def search(self, user_id, query, collection, topk=5):
        vec = self.embed.encode(query)
        QdrantStore.ensure_qdrant_collection(self.qdrant.client, collection, dim=1024 )
        res = self.qdrant.search(collection, vec, user_id, topk)

        return [r.payload["text"] for r in res]