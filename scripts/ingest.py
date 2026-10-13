"""CLI: ingest all PDFs in data/pdfs/ into Qdrant."""
from src.config import PDF_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.ingestion.indexer import index_chunks


def main():
    print(f"📂 Loading PDFs from {PDF_DIR}")
    pages = load_all_pdfs(PDF_DIR)
    if not pages:
        print("❌ No pages loaded. Run scripts/download_samples.py first.")
        return
    print(f"\n📄 Loaded {len(pages)} pages total")

    print("\n✂️  Chunking...")
    chunks = chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks")

    print("\n🔢 Embedding + indexing into Qdrant...")
    n = index_chunks(chunks)
    print(f"\n✅ Indexed {n} chunks successfully")

    # Show a sample chunk to verify metadata
    print("\n📌 Sample chunk metadata:")
    sample = chunks[0]
    print(f"   chunk_id   : {sample.chunk_id}")
    print(f"   source     : {sample.source}")
    print(f"   page       : {sample.page_number}")
    print(f"   char range : {sample.char_start}–{sample.char_end}")
    print(f"   text[:120] : {sample.text[:120]}...")


if __name__ == "__main__":
    main()