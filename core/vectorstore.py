"""
Vector store
============
Thin wrapper around ChromaDB (a local, file-based vector database). Handles
storing chunk text + embeddings + metadata, and dense (semantic) retrieval.
"""

from typing import List, Dict, Any
import chromadb
from core.config import CHROMA_DB_PATH

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def add_chunks(collection_name: str, chunk_ids: List[str], chunks: List[str],
                embeddings: List[List[float]], source: str) -> None:
    collection = get_client().get_or_create_collection(collection_name)
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": source} for _ in chunks],
    )


def get_all_chunks(collection_name: str) -> Dict[str, Any]:
    """Returns every chunk currently stored in a collection (used to build
    the BM25 index and to inspect the knowledge base)."""
    try:
        collection = get_client().get_collection(collection_name)
    except Exception:
        return {"ids": [], "documents": [], "metadatas": []}
    result = collection.get(include=["documents", "metadatas"])
    return {
        "ids": result["ids"],
        "documents": result["documents"],
        "metadatas": result["metadatas"],
    }


def dense_search(collection_name: str, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
    """Returns the top_k nearest chunks by cosine distance."""
    try:
        collection = get_client().get_collection(collection_name)
    except Exception:
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "distances", "metadatas"],
    )


def list_collections() -> List[str]:
    return [c.name for c in get_client().list_collections()]


def delete_collection(collection_name: str) -> None:
    try:
        get_client().delete_collection(collection_name)
    except Exception:
        pass


def list_sources(collection_name: str) -> List[Dict[str, Any]]:
    """Groups stored chunks by their originating document ('source') and
    returns a summary for each -- this is what powers the sources sidebar."""
    data = get_all_chunks(collection_name)
    summary: Dict[str, Dict[str, Any]] = {}
    for doc_id, meta, text in zip(data["ids"], data["metadatas"], data["documents"]):
        source = (meta or {}).get("source", "unknown")
        if source not in summary:
            summary[source] = {"source": source, "chunk_count": 0, "preview": text[:140]}
        summary[source]["chunk_count"] += 1
    return list(summary.values())


def delete_source(collection_name: str, source: str) -> int:
    """Deletes every chunk belonging to a single source document. Returns
    the number of chunks removed."""
    try:
        collection = get_client().get_collection(collection_name)
    except Exception:
        return 0
    result = collection.get(where={"source": source})
    ids = result.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
