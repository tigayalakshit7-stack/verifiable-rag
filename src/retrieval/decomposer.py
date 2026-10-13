"""
Query Decomposer — splits a complex question into focused sub-queries.

Example:
  "How does Self-RAG differ from corrective RAG in handling errors?"
    ->
  [
    {"query": "Self-RAG methodology and error handling", "rationale": "..."},
    {"query": "Corrective RAG (CRAG) methodology", "rationale": "..."},
    {"query": "comparison of RAG error handling approaches", "rationale": "..."}
  ]
"""
import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm_with_fallbacks
from src.retrieval.types import SubQuery
from src.config import MAX_SUBQUERIES


DECOMPOSER_PROMPT = """You are a research query analyst. Given a user's question, decompose it into {max_n} or fewer focused sub-queries that together would retrieve all the information needed to answer it comprehensively.

Question: {query}

Rules:
- If the question is simple and factual, return ONE sub-query (the question itself).
- If the question compares, contrasts, or requires multiple concepts, break it into 2-{max_n} sub-queries.
- Each sub-query should be a standalone search query optimized for semantic retrieval.
- Do NOT include meta-words like "compare" or "difference" unless necessary — focus on the concepts.

Return ONLY valid JSON, no markdown:
[
  {{"query": "...", "rationale": "..."}},
  ...
]
"""


def decompose_query(query: str) -> List[SubQuery]:
    llm = get_llm_with_fallbacks(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(DECOMPOSER_PROMPT)
    chain = prompt | llm

    response = chain.invoke({"query": query, "max_n": MAX_SUBQUERIES})
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        data = json.loads(content)
        subqueries = [SubQuery(**item) for item in data[:MAX_SUBQUERIES]]
        if not subqueries:
            raise ValueError("empty")
        return subqueries
    except Exception as e:
        print(f"Decomposer parse error: {e}\nRaw: {content}")
        # Graceful fallback: treat the original query as the only sub-query
        return [SubQuery(query=query, rationale="Fallback: single-query mode")]