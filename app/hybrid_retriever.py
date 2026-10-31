"""Hybrid retrieval over the shared Chroma collection.

Dense retrieval handles semantic paraphrases. A separate corpus-wide BM25 pass preserves
exact names, dates, years, quantities, units, and identifiers. The two ranked lists are
fused after candidate generation, so lexical-only matches are not required to first appear
in the dense top-k set.
"""
from __future__ import annotations

import math
import re
from typing import Any

from app import config
from app.tools import init_vectorstore


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", (text or "").lower())


def _bm25_scores(query: str, documents: list[Any]) -> list[float]:
    terms = _tokens(query)
    if not terms or not documents:
        return [0.0] * len(documents)
    tokenized = [_tokens(getattr(doc, "page_content", "")) for doc in documents]
    lengths = [len(tokens) for tokens in tokenized]
    avgdl = sum(lengths) / max(len(lengths), 1)
    document_frequency: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    n = len(tokenized)
    scores: list[float] = []
    for tokens, length in zip(tokenized, lengths):
        frequencies: dict[str, int] = {}
        for term in tokens:
            frequencies[term] = frequencies.get(term, 0) + 1
        score = 0.0
        for term in terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denominator = frequency + config.RAG_HYBRID_BM25_K1 * (
                1 - config.RAG_HYBRID_BM25_B
                + config.RAG_HYBRID_BM25_B * length / max(avgdl, 1.0)
            )
            score += idf * (frequency * (config.RAG_HYBRID_BM25_K1 + 1)) / max(denominator, 1e-9)
        scores.append(score)
    return scores


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi <= lo:
        return {key: (1.0 if hi > 0 else 0.0) for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_content_hash") or metadata.get("id") or getattr(doc, "page_content", ""))


def retrieve_hybrid_documents(query: str, k: int | None = None) -> list[Any]:
    """Return top-k documents from independent dense + lexical retrieval streams."""
    top_k = k or config.RAG_TOP_K
    store = init_vectorstore()
    dense_hits = store.similarity_search_with_score(query, k=max(top_k, config.RAG_HYBRID_CANDIDATES))
    corpus = store.get(include=["documents", "metadatas"])
    documents = []
    for content, metadata in zip(corpus.get("documents") or [], corpus.get("metadatas") or []):
        from langchain_core.documents import Document
        documents.append(Document(page_content=content or "", metadata=metadata or {}))
    if not documents:
        return []

    by_key = {_key(doc): doc for doc in documents}
    dense_similarity: dict[str, float] = {}
    for doc, distance in dense_hits:
        dense_similarity[_key(doc)] = 1.0 / (1.0 + max(float(distance), 0.0))
    lexical_raw = _bm25_scores(query, documents)
    lexical_raw_by_key = {_key(doc): score for doc, score in zip(documents, lexical_raw)}

    dense_norm = _normalize(dense_similarity)
    lexical_norm = _normalize(lexical_raw_by_key)
    union_keys = set(dense_norm) | {key for key, score in lexical_raw_by_key.items() if score > 0}
    fused = {
        key: config.RAG_HYBRID_VECTOR_WEIGHT * dense_norm.get(key, 0.0)
        + config.RAG_HYBRID_LEXICAL_WEIGHT * lexical_norm.get(key, 0.0)
        for key in union_keys
    }
    ranked_keys = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [by_key[key] for key in ranked_keys]


def format_retrieved_documents(documents: list[Any]) -> list[str]:
    formatted: list[str] = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        citation = " | ".join(
            str(value) for value in (
                metadata.get("source", "local"),
                metadata.get("title", ""),
                metadata.get("section", ""),
                metadata.get("url", ""),
            ) if value
        )
        formatted.append((f"Source: {citation}\n\n" if citation else "") + doc.page_content)
    return formatted
