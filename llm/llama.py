import requests
from MY_RAG_V2.config import OLLAMA_URL, MODEL

class LLM:

    def chat(self, prompt):

        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        return r.json()["response"]