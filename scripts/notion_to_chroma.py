"""Sync Notion pages into the existing Chroma knowledge_base collection."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

from app.tools import _get_vectorstore

LOG = logging.getLogger("notion_to_chroma")
API_BASE = "https://api.notion.com/v1"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def page_id_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return re.sub(r"[^0-9a-fA-F]", "", path.split("/")[-1]) if path else ""


def rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        data = item.get(kind, {}) if kind else {}
        value = data.get("content") or data.get("name") or data.get("url") or ""
        if kind == "mention":
            mention = data.get("page") or data.get("database") or data.get("date") or {}
            value = mention.get("title") or mention.get("start") or value or "[mention]"
        parts.append(str(value))
    return "".join(parts).strip()


def flatten_property(value: Any) -> Any:
    if not isinstance(value, dict):
        return str(value) if value is not None else ""
    kind = value.get("type")
    data = value.get(kind, {}) if kind else {}
    if kind == "title":
        return rich_text(data)
    if kind == "rich_text":
        return rich_text(data)
    if kind in {"select", "status"}:
        return str((data or {}).get("name", ""))
    if kind == "multi_select":
        return ", ".join(x.get("name", "") for x in data if isinstance(x, dict))
    if kind == "date":
        return str((data or {}).get("start", ""))
    if kind == "checkbox":
        return bool(data)
    if kind in {"number", "url", "email", "phone_number"}:
        return data if isinstance(data, (str, int, float, bool)) else ""
    if kind == "people":
        return ", ".join(p.get("name") or p.get("id", "") for p in data if isinstance(p, dict))
    return ""


def properties_to_metadata(page: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in (page.get("properties") or {}).items():
        flattened = flatten_property(value)
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
        if key and flattened not in ("", None):
            if isinstance(flattened, (str, int, float, bool)):
                result[f"prop_{key}"] = flattened
    return result


def page_title(page: dict[str, Any]) -> str:
    for value in (page.get("properties") or {}).values():
        if isinstance(value, dict) and value.get("type") == "title":
            title = rich_text(value.get("title"))
            if title:
                return title
    title = page.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "Untitled Notion page"


@dataclass
class Segment:
    path: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    def add(self, value: str, block_id: str = "") -> None:
        value = value.strip()
        if value:
            self.parts.append(value)
            if block_id and block_id not in self.block_ids:
                self.block_ids.append(block_id)

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts).strip()


def block_text(block: dict[str, Any]) -> str:
    kind = block.get("type", "")
    data = block.get(kind, {}) if kind else {}
    if not isinstance(data, dict):
        return ""
    if kind in {"paragraph", "heading_1", "heading_2", "heading_3", "heading_4",
                "bulleted_list_item", "numbered_list_item", "quote", "callout", "toggle", "to_do"}:
        value = rich_text(data.get("rich_text", []))
        if kind == "to_do":
            value = ("[x] " if data.get("checked") else "[ ] ") + value
        if kind == "callout":
            icon = data.get("icon") or {}
            if isinstance(icon, dict):
                icon = icon.get("emoji") or (icon.get("external") or {}).get("url") or ""
            value = (str(icon) + " " if icon else "") + value
        return value
    if kind == "code":
        fence = chr(96) * 3
        language = data.get("language") or "text"
        return f"{fence}{language}\n{rich_text(data.get('rich_text', []))}\n{fence}"
    if kind in {"bookmark", "embed", "link_preview"}:
        return (rich_text(data.get("caption", [])) + " " + str(data.get("url") or "")).strip()
    if kind in {"child_page", "child_database"}:
        return str(data.get("title") or "")
    if kind == "equation":
        return str(data.get("expression") or "")
    if kind in {"file", "image", "pdf"}:
        info = data.get("file") or data.get("external") or {}
        return (rich_text(data.get("caption", [])) or info.get("url") or "").strip()
    return ""


class NotionAPI:
    def __init__(self, token: str, version: str, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Notion-Version": version,
            "Content-Type": "application/json",
        })
        self.timeout = timeout

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path if path.startswith("https://") else f"{API_BASE}/{path.lstrip('/')}"
        for attempt in range(5):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(float(response.headers.get("Retry-After", min(2 ** attempt, 20))))
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        raise RuntimeError("Notion API retries exhausted")

    def page(self, page_id: str) -> dict[str, Any]:
        return self.get(f"pages/{page_id}")

    def children(self, block_id: str) -> list[dict[str, Any]]:
        rows, cursor = [], None
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            result = self.get(f"blocks/{block_id}/children", params=params)
            rows.extend(result.get("results", []))
            if not result.get("has_more") or not result.get("next_cursor"):
                return rows
            cursor = result["next_cursor"]


class BlockParser:
    def __init__(self, client: NotionAPI):
        self.client = client

    def parse(self, page: dict[str, Any]) -> list[Segment]:
        self.segments = [Segment()]
        self.path: list[str] = []
        self._walk(self.client.children(page["id"]), 0)
        return [part for part in self.segments if part.text]

    def current(self) -> Segment:
        return self.segments[-1]

    def _walk(self, blocks: list[dict[str, Any]], depth: int) -> None:
        if depth > 20:
            LOG.warning("Maximum block nesting reached; truncating deeper blocks")
            return
        for block in blocks:
            kind = block.get("type", "")
            block_id = str(block.get("id", ""))
            if kind in {"heading_1", "heading_2", "heading_3", "heading_4"}:
                level = int(kind[-1])
                self.path = self.path[:level - 1] + [block_text(block)]
                self.segments.append(Segment(list(self.path), [block_id], []))
                continue
            if kind == "table":
                table_rows = []
                for row in self.client.children(block_id):
                    cells = [rich_text(c) for c in row.get("table_row", {}).get("cells", [])]
                    if any(cells):
                        table_rows.append(" | ".join(cells))
                self.current().add("\n".join(table_rows), block_id)
            else:
                self.current().add(block_text(block), block_id)
            if block.get("has_children"):
                self._walk(self.client.children(block_id), depth + 1)


def mcp_snapshot_pages(payload: Any) -> list[dict[str, Any]]:
    """Read the connected Notion MCP wrapper shape observed in workspace fetch results."""
    items = payload if isinstance(payload, list) else (
        payload.get("pages", []) if isinstance(payload, dict) and isinstance(payload.get("pages"), list)
        else [payload]
    )
    pages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        wrapped = item.get("structuredContent") or item.get("structured_content") or item
        text = wrapped.get("text") or item.get("text") or ""
        if not text and isinstance(item.get("content"), str):
            try:
                text = json.loads(item["content"]).get("text", "")
            except (ValueError, AttributeError):
                pass
        if not text:
            continue
        url = wrapped.get("url") or item.get("url") or ""
        raw_id = str(wrapped.get("id") or item.get("id") or "")
        # The connected MCP wrapper currently supplies a page URL in its id field.
        # Notion's own API IDs are stable UUIDs; parse the path component from this
        # observed wrapper shape rather than persisting a mutable URL as the ID.
        page_id = page_id_from_url(raw_id) if raw_id.startswith("http") else raw_id
        page_id = page_id or page_id_from_url(url)
        title = wrapped.get("title") or item.get("title") or "Untitled Notion page"
        match = re.search(r"<content>(.*?)</content>", text, re.S)
        body = match.group(1) if match else text
        body = re.sub(r"<database\b[^>]*>.*?</database>", "\n", body, flags=re.S)
        body = re.sub(r"</?(?:toggle|quote|callout|content|page|p)(?:\s+[^>]*)?>", "\n", body)
        body = re.sub(r"<[^>]+>", "", body)
        pages.append({
            "id": str(page_id), "title": str(title), "url": str(url),
            "body_markdown": html.unescape(body).strip(),
            "last_edited_time": wrapped.get("page_last_edited_at") or "",
        })
    return pages


def markdown_segments(body: str) -> list[Segment]:
    result: list[Segment] = []
    path: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            result.append(Segment(list(path), [], [text]))
            lines.clear()

    for line in body.splitlines():
        heading = re.match(r"^\s{0,3}(#{1,4})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level, label = len(heading.group(1)), heading.group(2).strip()
            path[:] = path[:level - 1] + [label]
        else:
            lines.append(line)
    flush()
    return result


def split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    atoms = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result, current = [], ""
    for atom in atoms or [text]:
        sentences = re.split(r"(?<=[.!?])\s+", atom) if len(atom) > limit else [atom]
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    result.append(current)
                    current = ""
                result.extend(sentence[i:i + limit] for i in range(0, len(sentence), limit))
                continue
            candidate = (current + "\n\n" + sentence).strip() if current else sentence
            if len(candidate) > limit and current:
                result.append(current)
                current = sentence
            else:
                current = candidate
    if current:
        result.append(current)
    return result


def make_chunks(title: str, segments: list[Segment], max_chars: int, overlap_chars: int):
    rows = []
    for segment in segments:
        section = " > ".join(segment.path) or "Overview"
        prefix = f"Document: {title}\nSection: {section}\n\n"
        current = ""
        for atom in split_long_text(segment.text, max(300, max_chars - len(prefix))):
            candidate = (current + "\n\n" + atom).strip() if current else atom
            if len(prefix) + len(candidate) > max_chars and current:
                rows.append((prefix + current, section, ",".join(segment.block_ids)))
                overlap = current[-overlap_chars:].strip() if overlap_chars else ""
                current = (overlap + "\n\n" + atom).strip() if overlap else atom
            else:
                current = candidate
        if current:
            rows.append((prefix + current, section, ",".join(segment.block_ids)))
    return rows


def sync_page(vectorstore: Any, page: dict[str, Any], segments: list[Segment],
              max_chars: int = 2800, overlap_chars: int = 320) -> dict[str, Any]:
    page_id = str(page.get("id") or "")
    if not page_id:
        raise ValueError("Missing Notion page ID; stable IDs cannot be generated")
    title = page_title(page)
    url = page.get("url") or f"https://www.notion.so/{page_id.replace('-', '')}"
    last_edited = str(page.get("last_edited_time") or page.get("last_edited_at") or "")
    props = properties_to_metadata(page)
    body = "\n\n".join(segment.text for segment in segments)
    page_hash = digest(title + "\n" + body + json.dumps(props, sort_keys=True, default=str))
    existing = vectorstore.get(
        where={"$and": [{"source": "notion"}, {"notion_page_id": page_id}]},
        include=["metadatas"],
    )
    old_ids = list(existing.get("ids") or [])
    old_meta = existing.get("metadatas") or []
    if old_ids and old_meta and all(
        meta.get("page_content_hash") == page_hash for meta in old_meta if isinstance(meta, dict)
    ):
        return {"page_id": page_id, "title": title, "status": "unchanged", "chunks": len(old_ids)}

    chunks = make_chunks(title, segments, max_chars, overlap_chars)
    if not chunks:
        LOG.warning("No text found on %s; preserving any existing vectors", title)
        return {"page_id": page_id, "title": title, "status": "empty_skipped", "chunks": 0}

    docs, ids = [], []
    for index, (content, section, block_ids) in enumerate(chunks):
        key = "notion-" + digest(f"{page_id}|{section}|{digest(content)}")[:40]
        metadata = {
            "source": "notion", "content_type": "documentation",
            "notion_page_id": page_id, "title": title, "section": section,
            "url": url, "block_ids": block_ids, "last_edited_time": last_edited,
            "page_content_hash": page_hash, "chunk_content_hash": digest(content),
            "chunk_index": index,
        }
        metadata.update(props)
        docs.append(Document(page_content=content, metadata=metadata))
        ids.append(key)

    vectorstore.add_documents(docs, ids=ids)
    stale = sorted(set(old_ids) - set(ids))
    if stale:
        vectorstore.delete(ids=stale)
    return {"page_id": page_id, "title": title, "status": "synced",
            "chunks": len(ids), "stale_removed": len(stale)}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Sync Notion knowledge pages into existing Chroma")
    parser.add_argument("--page-ids", default=os.getenv("NOTION_PAGE_IDS", ""))
    parser.add_argument("--snapshot", type=Path,
                        help="JSON exported from the connected Notion fetch tool")
    parser.add_argument("--max-chars", type=int, default=2800)
    parser.add_argument("--overlap-chars", type=int, default=320)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")
    store = _get_vectorstore()
    output = []
    if args.snapshot:
        payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
        for page in mcp_snapshot_pages(payload):
            segments = markdown_segments(page.get("body_markdown", ""))
            output.append(sync_page(store, page, segments, args.max_chars, args.overlap_chars))
    else:
        token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
        if not token:
            parser.error("Set NOTION_TOKEN or pass --snapshot.")
        page_ids = [item.strip() for item in args.page_ids.split(",") if item.strip()]
        if not page_ids:
            parser.error("Pass --page-ids or set NOTION_PAGE_IDS.")
        client = NotionAPI(token, os.getenv("NOTION_VERSION", "2026-03-11"))
        normalizer = BlockParser(client)
        for page_id in page_ids:
            page = client.page(page_id)
            parent_type = (page.get("parent") or {}).get("type")
            if parent_type in {"database_id", "data_source_id"}:
                LOG.info("Skipping database record page %s; query its structured properties instead", page_id)
                continue
            output.append(sync_page(store, page, normalizer.parse(page),
                                    args.max_chars, args.overlap_chars))
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
