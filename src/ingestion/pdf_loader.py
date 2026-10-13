"""
PDF loader — extracts text page-by-page from PDFs.
Preserves page numbers so downstream chunks can cite exact locations.
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class PageContent:
    source: str          # filename
    page_number: int     # 1-indexed
    text: str


def load_pdf(path: Path) -> List[PageContent]:
    """Extract text from every page of a PDF."""
    reader = PdfReader(str(path))
    pages: List[PageContent] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  ⚠️  Page {i} extraction failed: {e}")
            text = ""
        text = text.strip()
        if text:
            pages.append(PageContent(
                source=path.name,
                page_number=i,
                text=text,
            ))
    return pages


def load_all_pdfs(pdf_dir: Path) -> List[PageContent]:
    """Load every PDF in a directory."""
    all_pages: List[PageContent] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {pdf_dir}")
        return all_pages

    for pdf_path in pdf_files:
        try:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"  ✅ {pdf_path.name}: {len(pages)} pages")
        except Exception as e:
            print(f"  ❌ {pdf_path.name}: {e}")
    return all_pages