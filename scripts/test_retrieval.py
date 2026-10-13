"""Test the retrieval engine in both modes and compare."""
from src.retrieval.retriever import retrieve


TEST_QUESTIONS = [
    # Simple factual — should stay as 1 sub-query in multi_hop
    "What is Retrieval-Augmented Generation?",

    # Comparative — should decompose into multiple sub-queries
    "How does Self-RAG differ from Corrective RAG in handling retrieval errors?",

    # Multi-concept — requires cross-document reasoning
    "What techniques do modern RAG systems use to improve factual accuracy?",
]


def show_result(result, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Query       : {result.original_query}")
    print(f"  Mode        : {result.mode}")
    print(f"  Sub-queries : {len(result.subqueries)}")
    for i, sq in enumerate(result.subqueries, 1):
        print(f"    [{i}] {sq.query}")
    print(f"  Candidates  : {result.total_candidates}")
    print(f"  After dedup : {result.final_count}")
    print(f"\n  Top 3 chunks:")
    for i, chunk in enumerate(result.chunks[:3], 1):
        snippet = chunk.text[:140].replace("\n", " ")
        sources = ", ".join(chunk.matched_subqueries[:2]) or "—"
        print(f"    [{i}] score={chunk.score:.3f}  {chunk.source} p.{chunk.page_number}")
        print(f"        {snippet}...")
        print(f"        matched: {sources}")


def main():
    for q in TEST_QUESTIONS:
        # Baseline
        baseline = retrieve(q, mode="single")
        show_result(baseline, "BASELINE (single mode)")

        # Multi-hop
        multihop = retrieve(q, mode="multi_hop")
        show_result(multihop, "MULTI-HOP (our approach)")

        # Comparison summary
        print(f"\n  📊 Baseline retrieved {baseline.final_count} unique chunks")
        print(f"  📊 Multi-hop retrieved {multihop.final_count} unique chunks")
        delta = multihop.final_count - baseline.final_count
        sign = "+" if delta >= 0 else ""
        print(f"  📊 Delta: {sign}{delta}")


if __name__ == "__main__":
    main()