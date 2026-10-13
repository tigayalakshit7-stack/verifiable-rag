"""
Embeddings wrapper using fastembed (ONNX-based, no torch dependency).
Wraps fastembed in a LangChain-compatible interface.
"""
from typing import List
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from src.config import EMBEDDING_MODEL


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around fastembed."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(next(iter(self.model.query_embed(text))))


# Singleton — loaded once per process
_embeddings = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings()
    return _embeddings