"""
Answer Synthesizer.

Given a question and retrieved chunks, produce a structured answer where
every claim is grounded in specific chunks with traceable citations.
"""
import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm_with_fallbacks
from src.retrieval.types import RetrievedChunk, RetrievalResult
from src.generation.types import Answer, Claim
from src.generation.citations import (
    build_chunk_lookup,
    validate_and_clean_claims,
    build_citations,
)


SYNTHESIZER_PROMPT = """You are a rigorous research assistant. Answer the user's question using ONLY the provided source chunks.

Question: {question}

Source chunks:
{chunks}

INSTRUCTIONS:
1. Break your answer into 3-6 discrete factual claims.
2. For each claim, cite the chunk IDs that support it (use the exact chunk_id strings).
3. Do NOT invent information not present in the chunks.
4. If the chunks don't contain enough information to answer, set "insufficient_evidence" to true and explain in "notes".
5. Every claim must have at least one valid chunk_id citation.

Return ONLY valid JSON in this exact shape (no markdown fences):
{{
  "claims": [
    {{"text": "...", "cited_chunk_ids": ["chunk_id_1", "chunk_id_2"]}}
  ],
  "insufficient_evidence": false,
  "notes": ""
}}
"""


def _format_chunks(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[chunk_id={c.chunk_id}] [source={c.source}] [page={c.page_number}] [score={c.score:.3f}]"
        parts.append(f"{header}\n{c.text}\n")
    return "\n---\n".join(parts)


def synthesize_answer(
    question: str,
    retrieval: RetrievalResult,
    max_chunks: int = 8,
) -> Answer:
    """Generate a claim-grounded answer from retrieved chunks."""
    # Use top N chunks to keep prompt size reasonable
    top_chunks = retrieval.chunks[:max_chunks]

    if not top_chunks:
        return Answer(
            question=question,
            claims=[],
            citations=[],
            insufficient_evidence=True,
            notes="No chunks were retrieved for this question.",
            mode=retrieval.mode,
        )

    llm = get_llm_with_fallbacks(temperature=0.1)
    prompt = ChatPromptTemplate.from_template(SYNTHESIZER_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "chunks": _format_chunks(top_chunks),
    })
    content = response.content.strip()

    # Strip markdown fences if the LLM added them
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        data = json.loads(content)
        raw_claims = [Claim(**c) for c in data.get("claims", [])]
        insufficient = bool(data.get("insufficient_evidence", False))
        notes = data.get("notes", "")
    except Exception as e:
        print(f"Synthesizer parse error: {e}\nRaw: {content[:500]}")
        return Answer(
            question=question,
            claims=[],
            citations=[],
            insufficient_evidence=True,
            notes=f"Failed to parse synthesizer output: {e}",
            mode=retrieval.mode,
        )

    # Validate + clean
    lookup = build_chunk_lookup(top_chunks)
    cleaned_claims = validate_and_clean_claims(raw_claims, lookup)
    citations = build_citations(cleaned_claims, lookup)

    # If every claim ended up unsupported, flag it
    if cleaned_claims and all(c.confidence == "low" for c in cleaned_claims):
        insufficient = True
        notes = (notes + " All claims lacked valid citations.").strip()

    return Answer(
        question=question,
        claims=cleaned_claims,
        citations=citations,
        insufficient_evidence=insufficient,
        notes=notes,
        mode=retrieval.mode,
    )