"""
Chunking
========
Uses LangChain's RecursiveCharacterTextSplitter to split raw text into
overlapping chunks suitable for embedding and retrieval.

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

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNK_SIZE, CHUNK_OVERLAP

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_text(text)