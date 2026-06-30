"""
Central configuration for the RAG pipeline.

Keeping every tunable value in one place makes it easy to experiment:
change CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME or HYBRID_ALPHA
and re-run ingestion to see how results change.
"""

import os

# ---- Chunking ----
# CHUNK_SIZE: max characters per chunk.
# CHUNK_OVERLAP: characters shared between consecutive chunks, so a sentence
# that gets cut at a chunk boundary still appears (partially) in the next chunk.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# ---- Embeddings ----
# A small, fast, local sentence-embedding model. No API key needed for this part.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ---- Retrieval ----
# HYBRID_ALPHA: weight given to dense (semantic) search vs sparse (keyword/BM25) search.
# 1.0  -> pure semantic search
# 0.0  -> pure keyword search
# 0.5  -> equal blend (default)
DEFAULT_HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 5))

# ---- Storage ----
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")

# ---- LLM (generation) ----
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
