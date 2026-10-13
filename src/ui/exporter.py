"""Export an Answer to clean Markdown for download."""
from src.generation.types import Answer


CONF = {"high": "HIGH", "medium": "MED", "low": "LOW"}


def answer_to_markdown(answer: Answer) -> str:
    lines = []
    lines.append(f"# {answer.question}\n")
    lines.append(f"_Mode: {answer.mode}_\n")

    if answer.insufficient_evidence:
        lines.append(f"> ⚠️ Insufficient evidence. {answer.notes}\n")

    lines.append("## Claims\n")
    for i, claim in enumerate(answer.claims, 1):
        refs = ", ".join(f"`{cid[:8]}`" for cid in claim.cited_chunk_ids) or "_no source_"
        lines.append(f"**{i}. [{CONF[claim.confidence]}]** {claim.text}")
        lines.append(f"   Sources: {refs}\n")

    lines.append("\n## Citations\n")
    for i, c in enumerate(answer.citations, 1):
        lines.append(f"**[{i}]** `{c.chunk_id[:8]}` · {c.source} · p.{c.page_number}")
        lines.append(f"> {c.snippet}\n")

    return "\n".join(lines)