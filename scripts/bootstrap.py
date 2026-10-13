"""Auto-bootstrap: download samples and ingest if collection is empty."""
import truststore
truststore.inject_into_ssl()

from src.vector_store import get_qdrant_client, get_collection_info
from src.config import PDF_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.ingestion.indexer import index_chunks


def bootstrap_if_empty() -> bool:
    """Returns True if ingestion ran, False if data already present."""
    try:
        info = get_collection_info(get_qdrant_client())
        if info.get("vectors_count", 0) > 0:
            return False
    except Exception:
        pass

    # Download if needed
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    if not list(PDF_DIR.glob("*.pdf")):
        from scripts.download_samples import main as download
        download()

    pages = load_all_pdfs(PDF_DIR)
    chunks = chunk_pages(pages)
    index_chunks(chunks)
    return True