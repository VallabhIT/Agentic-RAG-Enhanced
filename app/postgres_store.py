"""Optional PostgreSQL metadata store for application and ingestion state."""
from __future__ import annotations

import json
import os
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_sync (
    source TEXT NOT NULL,
    item_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_edited_time TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    sync_status TEXT NOT NULL DEFAULT 'synced',
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, item_id)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id UUID PRIMARY KEY,
    original_query TEXT NOT NULL,
    rewritten_query TEXT,
    route TEXT,
    status TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""


def get_connection():
    """Open a Psycopg 3 connection using DATABASE_URL."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install psycopg[binary] to use PostgreSQL") from exc
    return psycopg.connect(dsn)


def initialize_schema() -> None:
    # Execute DDL one statement at a time for compatibility with Psycopg's
    # parameterized/extended query protocol.
    with get_connection() as conn:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                conn.execute(statement)


def record_ingestion_sync(
    source: str,
    item_id: str,
    content_hash: str,
    chunk_count: int,
    last_edited_time: str | None = None,
    sync_status: str = "synced",
) -> None:
    """Upsert one source checkpoint after the corresponding vector upsert succeeds."""
    initialize_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_sync
                (source, item_id, content_hash, last_edited_time, chunk_count, sync_status, synced_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (source, item_id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                last_edited_time = EXCLUDED.last_edited_time,
                chunk_count = EXCLUDED.chunk_count,
                sync_status = EXCLUDED.sync_status,
                synced_at = NOW()
            """,
            (source, item_id, content_hash, last_edited_time, chunk_count, sync_status),
        )


def record_agent_run(
    run_id: str,
    original_query: str,
    rewritten_query: str | None,
    route: str | None,
    status: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist a compact audit record; do not store prompts/secrets or full document text."""
    initialize_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs
                (run_id, original_query, rewritten_query, route, status, finished_at, details)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s::jsonb)
            ON CONFLICT (run_id) DO UPDATE SET
                rewritten_query = EXCLUDED.rewritten_query,
                route = EXCLUDED.route,
                status = EXCLUDED.status,
                finished_at = NOW(),
                details = EXCLUDED.details
            """,
            (run_id, original_query, rewritten_query, route, status,
             json.dumps(details or {}, ensure_ascii=False)),
        )
