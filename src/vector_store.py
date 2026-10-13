"""
Qdrant local-mode client wrapper.

Uses Qdrant's embedded local mode — no server, no Docker.
Data persists on disk at data/qdrant_local/.
"""
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_DIM


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client in local persistent mode."""
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't already exist."""
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️  Collection already exists: {COLLECTION_NAME}")


def get_collection_info(client: QdrantClient) -> dict:
    """Return basic stats about the collection."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}