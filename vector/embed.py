from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def encode(self, text):
        return self.model.encode(text).tolist()