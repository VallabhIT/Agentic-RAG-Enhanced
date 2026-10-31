from __future__ import annotations

from types import SimpleNamespace

import app.hybrid_retriever as hybrid


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k=5):
        # Simulate a dense retriever that misses the exact numeric grocery match.
        return [(self.docs[1], 0.2)]

    def get(self, include=None):
        return {
            "documents": [doc.page_content for doc in self.docs],
            "metadatas": [doc.metadata for doc in self.docs],
        }


def test_hybrid_lexical_stream_recovers_exact_quantity(monkeypatch):
    docs = [
        SimpleNamespace(page_content="Weekend meal planning and pantry ideas.", metadata={"id": "a", "title": "Other"}),
        SimpleNamespace(page_content="Grocery list: rice 5 kg, cooking oil 2 L.", metadata={"id": "b", "title": "Grocery Shopping List"}),
    ]
    monkeypatch.setattr(hybrid, "init_vectorstore", lambda: FakeStore(docs))
    result = hybrid.retrieve_hybrid_documents("5 kg rice 2 L cooking oil", k=1)
    assert result[0].metadata["title"] == "Grocery Shopping List"
