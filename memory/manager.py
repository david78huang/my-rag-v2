from db.postgres import PostgresClient
from memory.sementic import SemanticMemory
from memory.extrator import MemoryExtractor
import logging
logger = logging.getLogger("uvicorn.error")


class MemoryManager:

    def __init__(self):

        self.pg = PostgresClient()
        self.semantic = SemanticMemory()
        self.extractor = MemoryExtractor()

    def store(self, user_id, text):

        mem = self.extractor.extract(text)
        logger.info("memory:", mem)

        # facts

        #for k, v in mem.get("facts", {}).items():
        #    self.pg.upsert(user_id, k, v)

        facts = mem.get("facts", {})
        if facts:
            self.pg.upsert_dict(user_id, facts)

        # semantic
        for s in mem.get("semantic", []):
            self.semantic.add(user_id, s)

