"""Server-side Supabase Postgres + pgvector storage for hosted RAG mode.

Run sql/supabase.sql in the Supabase SQL Editor before using this backend.
The database URL is used only on the Flask server, never sent to browsers.
"""

import math
from typing import Any, Dict, List

from core.config import SUPABASE_DB_URL, GEMINI_EMBEDDING_DIM, GEMINI_EMBEDDING_MODEL


def _connect():
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is missing. Add a Supabase Postgres connection URI.")
    import psycopg

    # The transaction pooler works with prepared statements disabled.
    return psycopg.connect(SUPABASE_DB_URL, sslmode="require",
                           prepare_threshold=None, connect_timeout=10)


def _vector(values: List[float]) -> str:
    if len(values) != GEMINI_EMBEDDING_DIM or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected a finite {GEMINI_EMBEDDING_DIM}-dimension embedding.")
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def add_chunks(notebook: str, ids: List[str], texts: List[str],
               embeddings: List[List[float]], source: str, fingerprint: str) -> None:
    if not (len(ids) == len(texts) == len(embeddings)):
        raise ValueError("Chunk text and embedding counts do not match.")
    rows = [
        (notebook, chunk_id, source, text, GEMINI_EMBEDDING_MODEL,
         _vector(embedding), fingerprint)
        for chunk_id, text, embedding in zip(ids, texts, embeddings)
    ]
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM public.notebook_chunks WHERE notebook = %s AND source = %s",
                           (notebook, source))
            if rows:
                # One INSERT avoids Psycopg's executemany pipeline, which the
                # Supabase transaction pooler does not support.
                values = ", ".join(
                    ["(%s, %s, %s, %s, %s, %s::extensions.vector, %s)"] * len(rows)
                )
                cursor.execute("""
                    INSERT INTO public.notebook_chunks
                      (notebook, chunk_id, source, content, embedding_model,
                       embedding, source_fingerprint)
                    VALUES """ + values, [value for row in rows for value in row])


def cached_source_count(notebook: str, source: str, fingerprint: str) -> int:
    with _connect() as connection:
        row = connection.execute("""
            SELECT COUNT(*) FROM public.notebook_chunks
            WHERE notebook = %s AND source = %s AND embedding_model = %s
              AND source_fingerprint = %s
        """, (notebook, source, GEMINI_EMBEDDING_MODEL, fingerprint)).fetchone()
    return row[0]


def get_all_chunks(notebook: str) -> Dict[str, Any]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT chunk_id, content, source FROM public.notebook_chunks
            WHERE notebook = %s AND embedding_model = %s ORDER BY chunk_id
        """, (notebook, GEMINI_EMBEDDING_MODEL)).fetchall()
    return {
        "ids": [row[0] for row in rows],
        "documents": [row[1] for row in rows],
        "metadatas": [{"source": row[2]} for row in rows],
    }


def dense_search(notebook: str, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT chunk_id, content, source,
                   embedding OPERATOR(extensions.<=>) %s::extensions.vector AS distance
            FROM public.notebook_chunks
            WHERE notebook = %s AND embedding_model = %s
            ORDER BY distance LIMIT %s
        """, (_vector(query_embedding), notebook, GEMINI_EMBEDDING_MODEL, top_k)).fetchall()
    return {
        "ids": [[row[0] for row in rows]],
        "documents": [[row[1] for row in rows]],
        "metadatas": [[{"source": row[2]} for row in rows]],
        "distances": [[row[3] for row in rows]],
    }


def list_collections() -> List[str]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT DISTINCT notebook FROM public.notebook_chunks
            WHERE embedding_model = %s ORDER BY notebook
        """, (GEMINI_EMBEDDING_MODEL,)).fetchall()
    return [row[0] for row in rows]


def delete_collection(notebook: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM public.notebook_chunks WHERE notebook = %s",
                           (notebook,))


def delete_source(notebook: str, source: str) -> int:
    with _connect() as connection:
        cursor = connection.execute("""
            DELETE FROM public.notebook_chunks
            WHERE notebook = %s AND source = %s
        """, (notebook, source))
        return cursor.rowcount
