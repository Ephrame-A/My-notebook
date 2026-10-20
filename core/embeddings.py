"""Embedding providers: the original local model and hosted Gemini API."""

import os
from typing import List
from core.config import EMBEDDING_MODEL_NAME, GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_DIM, RAG_MODE

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] Loading model '{EMBEDDING_MODEL_NAME}' (first call only)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _gemini_client(api_key: str = None):
    from google import genai

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("Enter a Gemini API key before uploading or asking a question.")
    return genai.Client(api_key=resolved_key)


def embed_texts(texts: List[str], api_key: str = None, source: str = None) -> List[List[float]]:
    """Embeds a list of strings into a list of float vectors."""
    if not texts:
        return []
    if RAG_MODE == "hosted":
        from google.genai import types

        client = _gemini_client(api_key)
        result = []
        # Separate Content objects return separate vectors in one request.
        # Small groups keep free-tier request usage and upload latency down.
        for start in range(0, len(texts), 16):
            group = texts[start:start + 16]
            contents = [
                types.Content(parts=[types.Part.from_text(
                    text=f"title: {source or 'none'} | text: {text}"
                )]) for text in group
            ]
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=GEMINI_EMBEDDING_DIM),
            )
            if len(response.embeddings or []) != len(group):
                raise RuntimeError("Gemini returned fewer embeddings than passages. Please try again.")
            result.extend(_values(embedding) for embedding in response.embeddings)
        return result
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def _single_vector(response) -> List[float]:
    embeddings = response.embeddings or []
    if len(embeddings) != 1:
        raise RuntimeError("Gemini returned no embedding. Please try again.")
    return _values(embeddings[0])


def _values(embedding) -> List[float]:
    values = list(embedding.values or [])
    if len(values) != GEMINI_EMBEDDING_DIM:
        raise RuntimeError("Gemini returned an unexpected embedding size.")
    return values


def embed_query(query: str, api_key: str = None) -> List[float]:
    """Convenience wrapper for embedding a single query string."""
    if RAG_MODE == "hosted":
        from google.genai import types

        response = _gemini_client(api_key).models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=f"task: question answering | query: {query}",
            config=types.EmbedContentConfig(output_dimensionality=GEMINI_EMBEDDING_DIM),
        )
        return _single_vector(response)
    return embed_texts([query])[0]
