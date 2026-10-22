import llm_clients
from llm_clients import *

POSTGRES = {
    "host": "localhost",
    "port": 5432,
    "database": "memory",
    "user": "postgres",
    "password": "Passw0rd1"
}

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

OLLAMA_URL = llm_clients.LLM_HOST
MODEL = llm_clients.LLM_MODEL

EMBED_MODEL = "BAAI/bge-m3"