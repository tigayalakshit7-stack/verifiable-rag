"""
Report generator — converts a list of AnswerMetrics into a markdown report
that can be committed to the repo or displayed in the UI.
"""
from typing import List, Dict
from datetime import datetime
from src.evaluation.metrics import AnswerMetrics


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _aggregate(metrics: List[AnswerMetrics]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "questions": n,
        "avg_claims": sum(m.claim_count for m in metrics) / n,
        "avg_citations": sum(m.citation_count for m in metrics) / n,
        "avg_grounding_rate": sum(m.grounding_rate for m in metrics) / n,
        "avg_high_conf_rate": sum(m.high_confidence_rate for m in metrics) / n,
        "avg_cross_source_rate": sum(m.cross_source_rate for m in metrics) / n,
        "avg_source_diversity": sum(m.source_diversity for m in metrics) / n,
        "avg_citation_density": sum(m.citation_density for m in metrics) / n,
        "avg_latency": sum(m.latency_seconds for m in metrics) / n,
        "insufficient_rate": sum(1 for m in metrics if m.insufficient_evidence) / n,
    }


def render_comparison_markdown(
    baseline_metrics: List[AnswerMetrics],
    multihop_metrics: List[AnswerMetrics],
) -> str:
    b = _aggregate(baseline_metrics)
    m = _aggregate(multihop_metrics)

    lines = []
    lines.append("# RAG Evaluation Report\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Baseline (single) | Multi-Hop (ours) | Δ |")
    lines.append("| :--- | ---: | ---: | ---: |")

    def row(label, key, fmt="num"):
        bv = b.get(key, 0)
        mv = m.get(key, 0)
        delta = mv - bv
        sign = "+" if delta >= 0 else ""
        if fmt == "pct":
            lines.append(f"| {label} | {_pct(bv)} | {_pct(mv)} | {sign}{_pct(delta)} |")
        elif fmt == "2f":
            lines.append(f"| {label} | {_fmt(bv)} | {_fmt(mv)} | {sign}{_fmt(delta)} |")
        else:
            lines.append(f"| {label} | {bv:.1f} | {mv:.1f} | {sign}{delta:.1f} |")

    row("Questions evaluated", "questions")
    row("Avg claims per answer", "avg_claims", "2f")
    row("Avg citations per answer", "avg_citations", "2f")
    row("Citation density (citations/claim)", "avg_citation_density", "2f")
    row("Grounding rate", "avg_grounding_rate", "pct")
    row("High-confidence rate", "avg_high_conf_rate", "pct")
    row("**Cross-source rate**", "avg_cross_source_rate", "pct")
    row("Avg source diversity (PDFs cited)", "avg_source_diversity", "2f")
    row("Insufficient-evidence rate", "insufficient_rate", "pct")
    row("Avg latency (s)", "avg_latency", "2f")

    lines.append("\n## Per-Question Breakdown\n")
    lines.append("| Question | Mode | Claims | Grounding | High-Conf | Cross-Src | Sources | Latency |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for bm, mm in zip(baseline_metrics, multihop_metrics):
        q = bm.question[:50] + ("..." if len(bm.question) > 50 else "")
        for entry in (bm, mm):
            lines.append(
                f"| {q} | {entry.mode} | {entry.claim_count} | "
                f"{_pct(entry.grounding_rate)} | {_pct(entry.high_confidence_rate)} | "
                f"{_pct(entry.cross_source_rate)} | {entry.source_diversity} | "
                f"{entry.latency_seconds:.2f}s |"
            )

    return "\n".join(lines)