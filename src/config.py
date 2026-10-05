"""Shared settings for ingestion and question answering."""
import os
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "data/pdfs"
CHROMA_DIR = "chroma_db"
COLLECTION = "tax_forms"

# Free, local embedding model (no API key needed).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ~1200 characters keeps most form-line instructions in one chunk.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

TOP_K = 6
LLM_MODEL = os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b")