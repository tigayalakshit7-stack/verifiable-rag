"""Download sample arxiv PDFs about RAG."""
import truststore
truststore.inject_into_ssl()  # use Windows cert store — fixes SSL on Python 3.14

import urllib.request
from src.config import PDF_DIR

SAMPLES = {
    "rag_original.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "self_rag.pdf": "https://arxiv.org/pdf/2310.11511.pdf",
    "corrective_rag.pdf": "https://arxiv.org/pdf/2401.15884.pdf",
}


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        dest = PDF_DIR / name
        if dest.exists():
            print(f"⏭️  Already exists: {name}")
            continue
        print(f"⬇️  Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"   ✅ {name} ({size_kb} KB)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    main()