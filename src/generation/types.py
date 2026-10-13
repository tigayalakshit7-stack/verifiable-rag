from pydantic import BaseModel, Field
from typing import List, Literal


class Claim(BaseModel):
    """A single factual claim with its supporting citations."""
    text: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"


class Citation(BaseModel):
    """A rendered citation for display."""
    chunk_id: str
    source: str
    page_number: int
    snippet: str


class Answer(BaseModel):
    """Final synthesized answer with structured claims and citations."""
    question: str
    claims: List[Claim]
    citations: List[Citation]
    insufficient_evidence: bool = False
    notes: str = ""
    mode: str = "multi_hop"