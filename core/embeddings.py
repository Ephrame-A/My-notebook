"""
Embeddings
==========
Turns text into dense numeric vectors using a local sentence-transformer
model (no API key / internet calls needed at query time, after the model
is downloaded once).

I load the model lazily and only once (singleton pattern) because loading
it is the slowest part of startup.
"""

from typing import List
from core.config import EMBEDDING_MODEL_NAME

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] Loading model '{EMBEDDING_MODEL_NAME}' (first call only)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a list of strings into a list of float vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([query])[0]
