"""
Multi-Hop Retriever.

Modes:
- 'single'    : baseline RAG — one search on the original query
- 'multi_hop' : decompose -> search each sub-query -> dedupe -> merge scores

Both modes hit the same Qdrant collection; the difference is in how
the query is issued and how results are combined.
"""
from typing import List, Dict
from src.embeddings import get_embeddings
from src.vector_store import get_qdrant_client
from src.retrieval.decomposer import decompose_query
from src.retrieval.types import (
    SubQuery, RetrievedChunk, RetrievalResult
)
from src.config import COLLECTION_NAME, TOP_K_PER_SUBQUERY


def _search_qdrant(query_vector: List[float], top_k: int) -> List[dict]:
    """Run a single Qdrant search and return the raw hits as dicts."""
    client = get_qdrant_client()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "text": h.payload["text"],
            "source": h.payload["source"],
            "page_number": h.payload["page_number"],
            "char_start": h.payload["char_start"],
            "char_end": h.payload["char_end"],
            "score": float(h.score),
        }
        for h in hits
    ]


def _dedupe_and_merge(
    all_hits: List[Dict],
    subquery_texts: List[str],
) -> List[RetrievedChunk]:
    """
    Group hits by chunk_id, keep the highest score,
    and record which sub-queries matched this chunk.
    """
    merged: Dict[str, RetrievedChunk] = {}

    for hit in all_hits:
        cid = hit["chunk_id"]
        if cid in merged:
            existing = merged[cid]
            # Keep highest score
            if hit["score"] > existing.score:
                existing.score = hit["score"]
            # Track which sub-query matched
            sq = hit.get("_subquery", "")
            if sq and sq not in existing.matched_subqueries:
                existing.matched_subqueries.append(sq)
        else:
            merged[cid] = RetrievedChunk(
                chunk_id=cid,
                text=hit["text"],
                source=hit["source"],
                page_number=hit["page_number"],
                char_start=hit["char_start"],
                char_end=hit["char_end"],
                score=hit["score"],
                matched_subqueries=[hit.get("_subquery", "")]
                if hit.get("_subquery") else [],
            )

    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


def retrieve(
    query: str,
    mode: str = "multi_hop",
    top_k: int = TOP_K_PER_SUBQUERY,
) -> RetrievalResult:
    """Run retrieval in either single-query or multi-hop mode."""
    embeddings = get_embeddings()

    # ---- Decompose (or skip) ----
    if mode == "single":
        subqueries = [SubQuery(query=query, rationale="Baseline single-query mode")]
    elif mode == "multi_hop":
        subqueries = decompose_query(query)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- Retrieve for each sub-query ----
    all_hits: List[Dict] = []
    for sq in subqueries:
        vec = embeddings.embed_query(sq.query)
        hits = _search_qdrant(vec, top_k)
        for h in hits:
            h["_subquery"] = sq.query
        all_hits.extend(hits)

    # ---- Dedupe & merge ----
    merged = _dedupe_and_merge(all_hits, [sq.query for sq in subqueries])

    return RetrievalResult(
        original_query=query,
        mode=mode,
        subqueries=subqueries,
        chunks=merged,
        total_candidates=len(all_hits),
        final_count=len(merged),
    )