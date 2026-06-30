"""
Hybrid retrieval
================
Combines two retrieval strategies and fuses their scores:

- Dense (semantic) search via ChromaDB: finds chunks that mean something
  similar to the query, even with different wording.
- Sparse (keyword) search via BM25: finds chunks that share exact terms
  with the query (great for IDs, names, rare/technical words).

Score fusion: each retriever's scores are normalized to [0, 1], then
combined with `alpha`:

    final_score = alpha * dense_score + (1 - alpha) * sparse_score

alpha=1.0 -> pure semantic, alpha=0.0 -> pure keyword, alpha=0.5 -> blend.
"""

from typing import List, Dict, Any
from core.embeddings import embed_query
from core.bm25 import build_bm25_index
from core import vectorstore
from core.config import DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def retrieve(collection_name: str, query: str, top_k: int = DEFAULT_TOP_K,
             alpha: float = DEFAULT_HYBRID_ALPHA) -> List[Dict[str, Any]]:
    """Runs hybrid retrieval and returns a ranked list of chunks with score
    breakdowns, so the UI can show *why* each chunk was retrieved."""

    all_chunks = vectorstore.get_all_chunks(collection_name)
    doc_ids, documents = all_chunks["ids"], all_chunks["documents"]
    text_by_id = dict(zip(doc_ids, documents))

    if not doc_ids:
        return []

    # --- Dense (semantic) scores ---
    query_embedding = embed_query(query)
    dense_raw = vectorstore.dense_search(collection_name, query_embedding, top_k=max(top_k * 3, top_k))
    dense_scores: Dict[str, float] = {}
    if dense_raw["ids"] and dense_raw["ids"][0]:
        for doc_id, distance in zip(dense_raw["ids"][0], dense_raw["distances"][0]):
            # Smaller cosine distance = more similar; convert to a similarity score.
            dense_scores[doc_id] = 1.0 / (1.0 + distance)

    # --- Sparse (keyword / BM25) scores ---
    bm25_index = build_bm25_index(doc_ids, documents)
    sparse_scores = dict(bm25_index.search(query, top_k=max(top_k * 3, top_k)))

    # --- Fuse ---
    dense_norm = _normalize(dense_scores)
    sparse_norm = _normalize(sparse_scores)
    candidate_ids = set(dense_norm) | set(sparse_norm)

    fused = []
    for doc_id in candidate_ids:
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        fused.append({
            "id": doc_id,
            "text": text_by_id.get(doc_id, ""),
            "dense_score": round(d, 4),
            "sparse_score": round(s, 4),
            "final_score": round(alpha * d + (1 - alpha) * s, 4),
        })

    fused.sort(key=lambda r: r["final_score"], reverse=True)
    return fused[:top_k]
