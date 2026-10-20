"""
Chunking
========
Uses LangChain locally or a small standard-library splitter in hosted mode
to split raw text into overlapping chunks suitable for retrieval.

Splitting strategy:
1. Try to split on paragraphs ("\n\n").
2. If a piece is still too large, split on newlines ("\n").
3. Then split on sentences (". ").
4. Then split on spaces (" ").
5. Finally, fall back to character-level splitting ("").

The resulting chunks have a configurable overlap to preserve context across
chunk boundaries.
"""

from typing import List

from core.config import CHUNK_SIZE, CHUNK_OVERLAP, RAG_MODE

# Ordered from coarsest to finest separator.
SEPARATORS = [
    "\n\n",  # Paragraphs
    "\n",    # Lines
    ". ",    # Sentences
    " ",     # Words
    "",      # Characters (fallback)
]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Args:
        text: Raw text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")

    if RAG_MODE == "hosted":
        return _chunk_text_hosted(text, chunk_size, overlap)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_text(text)


def _chunk_text_hosted(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Small dependency-free splitter for the hosted build."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_start = start + max(chunk_size // 2, overlap + 1)
            for separator in SEPARATORS[:-1]:
                boundary = text.rfind(separator, search_start, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
