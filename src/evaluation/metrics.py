"""
Evaluation metrics for RAG answers.

All metrics are computed from a (question, retrieval, answer) triple.
"""
from dataclasses import dataclass, field
from typing import List
from src.retrieval.types import RetrievalResult
from src.generation.types import Answer
from src.generation.citations import build_chunk_lookup


@dataclass
class AnswerMetrics:
    question: str
    mode: str

    # Basic counts
    claim_count: int = 0
    citation_count: int = 0

    # Grounding
    grounded_claims: int = 0        # claims with >= 1 valid citation
    high_confidence_claims: int = 0  # claims with >= 2 valid citations
    unsupported_claims: int = 0      # claims with 0 valid citations

    # Cross-source
    cross_source_claims: int = 0    # claims citing chunks from 2+ PDFs
    source_diversity: int = 0       # unique PDFs cited

    # Honesty
    insufficient_evidence: bool = False

    # Meta
    latency_seconds: float = 0.0
    notes: str = ""

    @property
    def grounding_rate(self) -> float:
        return self.grounded_claims / self.claim_count if self.claim_count else 0.0

    @property
    def high_confidence_rate(self) -> float:
        return self.high_confidence_claims / self.claim_count if self.claim_count else 0.0

    @property
    def cross_source_rate(self) -> float:
        return self.cross_source_claims / self.claim_count if self.claim_count else 0.0

    @property
    def citation_density(self) -> float:
        return self.citation_count / self.claim_count if self.claim_count else 0.0


def compute_metrics(
    retrieval: RetrievalResult,
    answer: Answer,
    latency: float,
) -> AnswerMetrics:
    """Compute all evaluation metrics for a single answer."""
    lookup = build_chunk_lookup(retrieval.chunks)

    m = AnswerMetrics(
        question=answer.question,
        mode=answer.mode,
        claim_count=len(answer.claims),
        citation_count=len(answer.citations),
        insufficient_evidence=answer.insufficient_evidence,
        latency_seconds=latency,
        notes=answer.notes,
    )

    cited_sources = set()

    for claim in answer.claims:
        n_valid = len([cid for cid in claim.cited_chunk_ids if cid in lookup])

        if n_valid == 0:
            m.unsupported_claims += 1
        else:
            m.grounded_claims += 1

        if n_valid >= 2:
            m.high_confidence_claims += 1

        # Check if cited chunks span multiple PDFs
        claim_sources = {
            lookup[cid].source
            for cid in claim.cited_chunk_ids
            if cid in lookup
        }
        cited_sources.update(claim_sources)
        if len(claim_sources) >= 2:
            m.cross_source_claims += 1

    m.source_diversity = len(cited_sources)
    return m