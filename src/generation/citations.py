"""
Citation validation — ensures the LLM only cites chunks that actually exist.

This prevents hallucinated citations (e.g., citing [abc123] when no such chunk was retrieved).
"""
from typing import Dict, List
from src.retrieval.types import RetrievedChunk
from src.generation.types import Claim, Citation


def build_chunk_lookup(chunks: List[RetrievedChunk]) -> Dict[str, RetrievedChunk]:
    """Map chunk_id -> RetrievedChunk for quick lookup."""
    return {c.chunk_id: c for c in chunks}


def validate_and_clean_claims(
    claims: List[Claim],
    chunk_lookup: Dict[str, RetrievedChunk],
) -> List[Claim]:
    """
    Remove hallucinated citations and recompute confidence based on
    how many VALID sources support each claim.
    """
    cleaned: List[Claim] = []

    for claim in claims:
        valid_ids = [cid for cid in claim.cited_chunk_ids if cid in chunk_lookup]
        n = len(valid_ids)

        if n >= 2:
            confidence = "high"
        elif n == 1:
            confidence = "medium"
        else:
            confidence = "low"

        cleaned.append(Claim(
            text=claim.text,
            cited_chunk_ids=valid_ids,
            confidence=confidence,
        ))

    return cleaned


def build_citations(
    claims: List[Claim],
    chunk_lookup: Dict[str, RetrievedChunk],
    snippet_len: int = 220,
) -> List[Citation]:
    """Build a deduplicated list of Citation objects from all cited chunks."""
    seen = set()
    citations: List[Citation] = []

    for claim in claims:
        for cid in claim.cited_chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunk_lookup[cid]
            snippet = chunk.text[:snippet_len].strip().replace("\n", " ")
            citations.append(Citation(
                chunk_id=cid,
                source=chunk.source,
                page_number=chunk.page_number,
                snippet=snippet + ("..." if len(chunk.text) > snippet_len else ""),
            ))

    return citations