"""
Renderer — converts an Answer into human-readable markdown for display.
Used by both the CLI test and the Streamlit UI later.
"""
from src.generation.types import Answer


CONFIDENCE_BADGE = {
    "high": "🟢 HIGH",
    "medium": "🟡 MED",
    "low": "🔴 LOW",
}


def render_answer_markdown(answer: Answer) -> str:
    lines = []
    lines.append(f"### ❓ {answer.question}\n")

    if answer.insufficient_evidence:
        lines.append("> ⚠️ **Insufficient evidence to answer confidently.**\n")
        if answer.notes:
            lines.append(f"> {answer.notes}\n")

    if answer.claims:
        lines.append("### 📝 Claims\n")
        for i, claim in enumerate(answer.claims, 1):
            badge = CONFIDENCE_BADGE[claim.confidence]
            refs = ", ".join(f"`{cid[:8]}`" for cid in claim.cited_chunk_ids) or "—"
            lines.append(f"**{i}. [{badge}]** {claim.text}")
            lines.append(f"   ↳ Sources: {refs}\n")

    if answer.citations:
        lines.append("\n### 📚 Source Citations\n")
        for i, c in enumerate(answer.citations, 1):
            lines.append(f"**[{i}]** `{c.chunk_id[:8]}` · {c.source} · p.{c.page_number}")
            lines.append(f"> {c.snippet}\n")

    return "\n".join(lines)