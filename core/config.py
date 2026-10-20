"""
Central configuration for the RAG pipeline.

Keeping every tunable value in one place makes it easy to experiment:
change CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME or HYBRID_ALPHA
and re-run ingestion to see how results change.
"""

import os

# The original local stack stays the default. Hosted mode uses Gemini
# embeddings and a Supabase Postgres database for a small Render instance.
RAG_MODE = os.getenv("RAG_MODE", "local").lower()
if RAG_MODE not in {"local", "hosted"}:
    raise ValueError("RAG_MODE must be 'local' or 'hosted'.")

# ---- Chunking ----
# CHUNK_SIZE: max characters per chunk.
# CHUNK_OVERLAP: characters shared between consecutive chunks, so a sentence
# that gets cut at a chunk boundary still appears (partially) in the next chunk.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# ---- Embeddings ----
# A small, fast, local sentence-embedding model. No API key needed for this part.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_EMBEDDING_DIM = 768  # Must match sql/supabase.sql.
MAX_HOSTED_CHUNKS_PER_FILE = int(os.getenv("MAX_HOSTED_CHUNKS_PER_FILE", 80))
MAX_HOSTED_SOURCES = int(os.getenv("MAX_HOSTED_SOURCES", 5))

# ---- Retrieval ----
# HYBRID_ALPHA: weight given to dense (semantic) search vs sparse (keyword/BM25) search.
# 1.0  -> pure semantic search
# 0.0  -> pure keyword search
# 0.5  -> equal blend (default)
DEFAULT_HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 5))

# ---- Storage ----
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")

# ---- LLM (generation) ----
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# ---- Audio generation ----
AUDIO_CACHE_DIR = os.getenv("AUDIO_CACHE_DIR", "./audio_cache")
AUDIO_VOICE_HOST_A = os.getenv("AUDIO_VOICE_HOST_A", "en-US-GuyNeural")
AUDIO_VOICE_HOST_B = os.getenv("AUDIO_VOICE_HOST_B", "en-US-JennyNeural")
AUDIO_VOICE_NARRATOR = os.getenv("AUDIO_VOICE_NARRATOR", "en-US-AriaNeural")
