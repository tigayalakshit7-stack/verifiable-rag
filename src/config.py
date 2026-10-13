import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model fallback chain (same as Project #1) ---
MODEL_PRIMARY = "openai/gpt-oss-120b"
MODEL_SECONDARY = "qwen/qwen3.8-27b"
MODEL_TERTIARY = "llama-3.3-70b-versatile"

# --- Embedding model (fastembed) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small outputs 384-dim vectors

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval ---
TOP_K_PER_SUBQUERY = 4
MAX_SUBQUERIES = 3

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = str(PROJECT_ROOT / "data" / "qdrant_local")
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

# --- Qdrant collection name ---
COLLECTION_NAME = "research_papers"