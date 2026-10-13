from pydantic import BaseModel, Field
from typing import List


class SubQuery(BaseModel):
    """A single decomposed sub-query with its reasoning."""
    query: str
    rationale: str


class RetrievedChunk(BaseModel):
    """A chunk retrieved from Qdrant with its score and metadata."""
    chunk_id: str
    text: str
    source: str
    page_number: int
    char_start: int
    char_end: int
    score: float
    matched_subqueries: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """The complete retrieval output — sub-queries + final ranked chunks."""
    original_query: str
    mode: str
    subqueries: List[SubQuery]
    chunks: List[RetrievedChunk]
    total_candidates: int   # before dedup
    final_count: int        # after dedup