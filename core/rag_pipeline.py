"""
RAG pipeline
============
Glues chunking -> embedding -> storage (ingestion) and
retrieval -> prompt construction -> LLM generation (answering) together.

The answering side is citation-aware: the LLM is asked to mark claims with
[1], [2]... referencing the numbered sources it was given, the same way
NotebookLM-style tools do. The frontend turns those markers into clickable
chips that open the matching source passage.
"""

import os
from typing import List, Dict, Any

from core.chunking import chunk_text
from core.embeddings import embed_texts
from core import vectorstore
from core.retriever import retrieve
from core import chat_history
from core.config import GEMINI_MODEL_NAME, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA


def ingest_document(text: str, source: str, collection_name: str,
                     chunk_size: int = None, chunk_overlap: int = None) -> int:
    """Chunks, embeds, and stores a document's text. Returns chunk count."""
    kwargs = {}
    if chunk_size:
        kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        kwargs["overlap"] = chunk_overlap

    chunks = chunk_text(text, **kwargs)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    chunk_ids = [f"{source}::chunk_{i}" for i in range(len(chunks))]

    vectorstore.add_chunks(collection_name, chunk_ids, chunks, embeddings, source)
    return len(chunks)


SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the \
numbered sources below -- never use outside knowledge. \
After every claim, cite the source(s) it came from like this: [1] or [2][3]. \
If the sources don't contain the answer, say so plainly instead of guessing. \
Be concise and direct."""


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], history_text: str) -> str:
    sources_block = "\n\n".join(
        f"[{i+1}] (from {c['id'].split('::')[0]})\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    parts = [SYSTEM_PROMPT, ""]
    if history_text:
        parts += [history_text, ""]
    parts += [
        "Sources:",
        sources_block,
        "",
        f"Question: {query}",
    ]
    return "\n".join(parts)


def answer_query(collection_name: str, query: str, top_k: int = DEFAULT_TOP_K,
                  alpha: float = DEFAULT_HYBRID_ALPHA, use_history: bool = True,
                  api_key: str = None) -> Dict[str, Any]:
    """Runs retrieval, builds a citation-aware prompt, calls the LLM, and
    records the turn in the notebook's conversation history."""
    retrieved = retrieve(collection_name, query, top_k=top_k, alpha=alpha)

    if not retrieved:
        return {
            "answer": "This notebook has no sources yet. Add a document to get started.",
            "retrieved_chunks": [],
        }

    history_text = chat_history.format_history_for_prompt(collection_name) if use_history else ""
    prompt = build_prompt(query, retrieved, history_text)

    try:
        answer_text = _call_llm(prompt, api_key=api_key)
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        answer_text = "I'm sorry, I was unable to generate an answer at this time (e.g., API limits reached or invalid key). Please see the retrieved sources in the citation inspector on the right."

    if use_history:
        chat_history.add_turn(collection_name, "user", query)
        chat_history.add_turn(collection_name, "assistant", answer_text)

    # Number the chunks to match the [1], [2]... citations used in the prompt/answer.
    numbered_chunks = [
        {**c, "citation_number": i + 1, "source": c["id"].split("::")[0]}
        for i, c in enumerate(retrieved)
    ]

    return {"answer": answer_text, "retrieved_chunks": numbered_chunks}


def _call_llm(prompt: str, api_key: str = None) -> str:
    from google import genai

    # Prefer the caller-supplied key; fall back to server env var.
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "No Gemini API key found. Please enter your API key in the settings panel."
        )

    client = genai.Client(api_key=resolved_key)
    response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return response.text
