from __future__ import annotations

from scripts.notion_to_chroma import (
    Segment,
    chunk_segments,
    markdown_segments,
    mcp_snapshot_pages,
    sync_page,
)


class FakeVectorStore:
    def __init__(self):
        self.rows = {}

    def get(self, where=None, include=None):
        page_id = where["$and"][1]["notion_page_id"]
        rows = [(key, value) for key, value in self.rows.items()
                if value["metadata"].get("notion_page_id") == page_id]
        return {
            "ids": [key for key, _ in rows],
            "metadatas": [value["metadata"] for _, value in rows],
        }

    def add_documents(self, docs, ids):
        for doc, key in zip(docs, ids):
            self.rows[key] = {"page_content": doc.page_content, "metadata": doc.metadata}

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)


def test_mcp_snapshot_adapter_reads_workspace_response_shape():
    wrapper = {
        "structuredContent": {
            "id": "https://app.notion.com/p/40913ff48027813f8a87e67b0c490e7d",
            "title": "Agentic RAG Notes",
            "url": "https://app.notion.com/p/40913ff48027813f8a87e67b0c490e7d",
            "text": (
                '<page><properties>{"title":"Agentic RAG Notes"}</properties>'
                "<content># Retrieval\n\nChunk by Notion headings.</content></page>"
            ),
            "page_last_edited_at": "2026-10-30T16:35:16.514Z",
        }
    }
    pages = mcp_snapshot_pages(wrapper)
    assert len(pages) == 1
    assert pages[0]["title"] == "Agentic RAG Notes"
    assert pages[0]["id"] == "40913ff48027813f8a87e67b0c490e7d"
    assert "Chunk by Notion headings." in pages[0]["body_markdown"]


def test_markdown_parser_preserves_heading_context():
    segments = markdown_segments("# RAG\nFirst paragraph.\n\n## Chunking\nChunk by headings.")
    assert [segment.path for segment in segments] == [["RAG"], ["RAG", "Chunking"]]
    chunks = chunk_segments("Notes", segments, max_chars=1000, overlap_chars=0)
    assert "Section: RAG" in chunks[0][0]
    assert "Section: RAG > Chunking" in chunks[1][0]


def test_sync_page_upserts_stable_chunks_and_skips_unchanged_page():
    store = FakeVectorStore()
    page = {
        "id": "page-123",
        "title": "Agentic RAG Notes",
        "url": "https://www.notion.so/page-123",
        "last_edited_time": "2026-10-30T10:00:00Z",
    }
    segments = [Segment(path=["RAG"], parts=["Chunk documents at heading boundaries."])]
    first = sync_page(store, page, segments)
    assert first["status"] == "synced"
    assert first["chunks"] == 1
    second = sync_page(store, page, segments)
    assert second["status"] == "unchanged"
    assert len(store.rows) == 1

    changed = [Segment(path=["RAG"], parts=["Use structure-aware chunking and sentence fallback."])]
    third = sync_page(store, page, changed)
    assert third["status"] == "synced"
    assert third["stale_removed"] == 1
    assert len(store.rows) == 1
    metadata = next(iter(store.rows.values()))["metadata"]
    assert metadata["source"] == "notion"
    assert metadata["notion_page_id"] == "page-123"
    assert metadata["section"] == "RAG"
    assert metadata["url"] == page["url"]


def test_oversized_sections_split_without_losing_text():
    text = "Sentence one. " * 80
    chunks = chunk_segments("Long page", [Segment(path=["Long"], parts=[text])],
                            max_chars=500, overlap_chars=40)
    assert len(chunks) > 1
    joined = " ".join(chunk[0] for chunk in chunks)
    assert "Sentence one." in joined
