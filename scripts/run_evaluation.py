"""
Evaluation harness CLI.

Runs a fixed test set through BOTH modes (single + multi_hop),
computes metrics, and writes a markdown report to docs/evaluation.md.
"""
import time
from pathlib import Path
from src.retrieval.retriever import retrieve
from src.generation.synthesizer import synthesize_answer
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import render_comparison_markdown
from src.config import PROJECT_ROOT


# A mixed test set: factual, comparative, multi-concept
TEST_SET = [
    "What is Retrieval-Augmented Generation?",
    "How does Self-RAG differ from Corrective RAG in handling retrieval errors?",
    "What techniques do modern RAG systems use to improve factual accuracy?",
    "Why is retrieval quality important for RAG performance?",
    "What role does the retriever play in RAG?",
]


def evaluate_one(question: str, mode: str):
    t0 = time.time()
    retrieval = retrieve(question, mode=mode)
    answer = synthesize_answer(question, retrieval)
    latency = time.time() - t0
    metrics = compute_metrics(retrieval, answer, latency)
    return metrics


def main():
    print(f"Running evaluation on {len(TEST_SET)} questions × 2 modes\n")

    baseline_metrics = []
    multihop_metrics = []

    for i, q in enumerate(TEST_SET, 1):
        print(f"[{i}/{len(TEST_SET)}] {q[:60]}")

        print("   → baseline...")
        baseline_metrics.append(evaluate_one(q, mode="single"))

        print("   → multi-hop...")
        multihop_metrics.append(evaluate_one(q, mode="multi_hop"))

    print("\n📝 Writing report to docs/evaluation.md")
    report = render_comparison_markdown(baseline_metrics, multihop_metrics)

    out_path = PROJECT_ROOT / "docs" / "evaluation.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"✅ Report written: {out_path}\n")
    print("=" * 72)
    print(report)


if __name__ == "__main__":
    main()