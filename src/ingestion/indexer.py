"""
Indexer — embeds chunks and upserts them into Qdrant.

Uses batching to keep memory low and throughput high.
Converts hex chunk IDs into Qdrant-compatible integer point IDs.
"""
from typing import List
from qdrant_client.models import PointStruct
from src.embeddings import get_embeddings
from src.vector_store import get_qdrant_client, ensure_collection
from src.ingestion.chunker import Chunk
from src.config import COLLECTION_NAME


def _chunk_to_point_id(chunk_id: str) -> int:
    """Qdrant accepts unsigned ints — convert the 16-hex-char ID."""
    return int(chunk_id, 16)


def index_chunks(chunks: List[Chunk], batch_size: int = 64) -> int:
    client = get_qdrant_client()
    ensure_collection(client)
    embeddings = get_embeddings()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embeddings.embed_documents([c.text for c in batch])

        points = [
            PointStruct(
                id=_chunk_to_point_id(c.chunk_id),
                vector=vec,
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source": c.source,
                    "page_number": c.page_number,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                },
            )
            for c, vec in zip(batch, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"  📥 Indexed {total}/{len(chunks)} chunks")

    return total