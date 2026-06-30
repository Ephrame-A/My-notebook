"""
Sparse retrieval: BM25
=======================
A from-scratch implementation of the BM25 ranking algorithm using a plain
inverted index. =

BM25 score for a term in a document combines:
- TF  (term frequency): how often the term appears in this document.
- IDF (inverse document frequency): how rare the term is across all documents
  (rare terms are more informative, so they're weighted higher).
- Document length normalization: long documents naturally contain more word
  occurrences, so raw TF is normalized against the average document length.
"""

import math
import re
from typing import List, Dict, Tuple


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # term frequency saturation: higher = TF matters more
        self.b = b    # length normalization strength: 0 = none, 1 = full
        self.inverted_index: Dict[str, List[str]] = {}   # term -> [doc_id, ...]
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: count}
        self.doc_lengths: Dict[str, int] = {}             # doc_id -> token count
        self.documents: Dict[str, str] = {}                # doc_id -> raw text
        self.total_docs = 0
        self.avg_doc_length = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return cleaned.split()

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        if not tokens:
            return

        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1

        for token in tokens:
            self.term_freqs.setdefault(token, {})
            self.term_freqs[token][doc_id] = self.term_freqs[token].get(doc_id, 0) + 1
        for token in set(tokens):
            self.inverted_index.setdefault(token, [])
            self.inverted_index[token].append(doc_id)

        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def _idf(self, term: str) -> float:
        df = len(self.inverted_index.get(term, []))
        if df == 0:
            return 0.0
        # +0.5 / +1.0 smoothing keeps the score well-behaved for very common terms.
        return math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Returns a list of (doc_id, score) sorted by descending BM25 score."""
        if self.total_docs == 0:
            return []

        scores: Dict[str, float] = {}
        for term in self._tokenize(query):
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            for doc_id in self.inverted_index[term]:
                tf = self.term_freqs[term][doc_id]
                doc_len = self.doc_lengths[doc_id]
                len_norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
                term_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * len_norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def build_bm25_index(doc_ids: List[str], texts: List[str]) -> BM25Index:
    """Builds a fresh in-memory BM25 index from a list of chunks.

    We rebuild this on demand from whatever is stored in Chroma rather than
    persisting it separately -- it's cheap for learning-scale datasets and
    keeps the sparse and dense stores trivially in sync.
    """
    index = BM25Index()
    for doc_id, text in zip(doc_ids, texts):
        index.add_document(doc_id, text)
    return index
