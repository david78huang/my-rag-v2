from fastapi import FastAPI
from db.postgres import PostgresClient
from memory.manager import MemoryManager
import logging
logger = logging.getLogger("uvicorn, error")
from query_mineru_qdrant import *
app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    pg = PostgresClient()
    mem = MemoryManager()
    result = ask(question=req.message, qdrant_url=req.qdrant_url, collection_name=req.collection, req=req, pg=pg, mem=mem)
    logger.info(result)
    return result
