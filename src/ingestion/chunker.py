"""
Chunker — splits page text into overlapping chunks with full metadata.

Each chunk keeps:
- deterministic ID (from source + page + offset)
- character offsets within the page
- source filename + page number
"""
import hashlib
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.pdf_loader import PageContent
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_number: int
    char_start: int
    char_end: int


def _make_chunk_id(source: str, page: int, start: int, text: str) -> str:
    """Deterministic ID so re-indexing the same PDF is idempotent."""
    key = f"{source}:{page}:{start}:{text[:80]}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks: List[Chunk] = []

    for page in pages:
        splits = splitter.split_text(page.text)
        cursor = 0

        for split in splits:
            # Locate the chunk within the page text
            start = page.text.find(split, cursor)
            if start == -1:
                # Overlap shifted things — reset to cursor
                start = cursor
            end = start + len(split)
            cursor = max(cursor, end)

            chunks.append(Chunk(
                chunk_id=_make_chunk_id(page.source, page.page_number, start, split),
                text=split,
                source=page.source,
                page_number=page.page_number,
                char_start=start,
                char_end=end,
            ))
    return chunks