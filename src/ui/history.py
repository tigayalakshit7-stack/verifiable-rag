"""Session-scoped history of past questions."""
from dataclasses import dataclass
from typing import List
import time


@dataclass
class HistoryEntry:
    question: str
    timestamp: float
    claim_count: int
    high_conf_count: int
    latency: float


def add_entry(history: List[HistoryEntry], question: str, answer, latency: float) -> None:
    if not question.strip():
        return
    history.append(HistoryEntry(
        question=question,
        timestamp=time.time(),
        claim_count=len(answer.claims),
        high_conf_count=sum(1 for c in answer.claims if c.confidence == "high"),
        latency=latency,
    ))
    if len(history) > 20:
        history.pop(0)


def format_relative(ts: float) -> str:
    delta = int(time.time() - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"