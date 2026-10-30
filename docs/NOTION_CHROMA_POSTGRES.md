# Notion ingestion and relational storage

This is an additive change. It keeps the existing LangGraph graph and the persistent Chroma collection already used by the RAG tools.

## What was inspected in the connected Notion workspace

The connected Notion integration returned these actual shapes:

- A page fetch wrapper included structuredContent.id, title, text, url, metadata, verification, and page_last_edited_at. The text field contained Notion-flavored Markdown inside a page wrapper, with a JSON properties block and content wrapper.
- A page containing a database included a database tag with a database URL and a collection:// data-source URL.
- Fetching that database returned a data-sources section, each data source's data-source-state JSON (typed schema and property options), a sqlite-table definition, and saved views.
- Querying the existing To Do data source returned JSON rows with Task name (title), Status (status), Due date (date), Assignee (person), and a page URL.

These entities are distinct. A page body and database rows are not the same shape and must not be parsed as if they were. The current task database has seven onboarding rows; the code intentionally does not embed these routine rows as prose.

The connected ChatGPT Notion session is not a local Python API credential. The script supports either the official Notion API using a separately created integration token, or JSON snapshots exported from the connected Notion fetch response.

## Local Notion access

1. Create an internal integration in Notion's integrations settings.
2. Give it read-content capability only for the intended project knowledge pages.
3. Share each chosen page with the integration. An integration token cannot read pages that have not been shared with it.
4. Copy .env.example to .env, set NOTION_TOKEN, and set NOTION_PAGE_IDS to comma-separated page IDs.
5. Never commit .env; restrict the integration to project pages rather than granting unnecessary workspace-wide access.

The ingestion script defaults to Notion API version 2026-03-11. Override NOTION_VERSION only when intentionally testing another supported version.

## Sync Notion into the existing Chroma store

Install dependencies and configure credentials:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    # edit .env; do not commit it

Run a live sync:

    python -m scripts.notion_to_chroma --page-ids YOUR_PAGE_ID,YOUR_OTHER_PAGE_ID

The script:
- calls GET /v1/pages/{page_id} and GET /v1/blocks/{block_id}/children;
- follows has_more and next_cursor for each child-block page;
- recursively reads child blocks and tables;
- normalizes rich text, checklists, code, headings, callouts, and common media captions/URLs;
- creates structure-aware chunks and includes title/section context in each chunk;
- reuses the existing app.tools vector-store factory, collection name knowledge_base, directory data/chroma, and existing sentence-transformer embedding function;
- stores page and section metadata with every vector;
- creates deterministic vector IDs, skips unchanged pages based on a page-level content hash, upserts current chunks, and removes stale chunks only for that Notion page after successful additions.

### Connected Notion MCP snapshot mode

When you do not want to create a local Notion API integration yet, save a page fetch result (or an array of fetch results) as JSON, preserving structuredContent and text fields. Run:

    python -m scripts.notion_to_chroma --snapshot notion_fetch_snapshot.json

The snapshot adapter reads the actual MCP wrapper keys observed above. In MCP mode the integration's structuredContent.id may be a Notion page URL rather than a UUID; that value remains a stable identifier in Chroma metadata. The live API path uses the Notion page UUID.

Database schemas and rows are not embedded by default. Keep task title, status, due date, and assignee as typed Notion/SQL properties. If a future database record has a substantial explanatory description worth searching semantically, add a specific description indexer and label it content_type=database_record; do not embed every routine status row.

## Chunking strategy: structure-first hybrid

Use structure-aware chunking first. A Notion heading defines a semantic section; include its heading path (for example, RAG > Chunking > Table handling) in the chunk so that an isolated paragraph still carries context.

Group adjacent paragraphs, list items, quotes, callouts, and code blocks that explain one idea. Keep a table's header and rows together when possible. For an oversized section, split at blank-line/paragraph/list boundaries; split a single oversized paragraph by sentences; use hard character slicing only as a last fallback. The script defaults to about 2,800 characters (roughly 700 tokens, depending on language/model) with 320 characters of overlap. Tune with retrieval evaluation rather than treating these numbers as universal constants.

Semantic chunking is the fallback, not the first operation. Notion already supplies useful structure; discovering boundaries from embeddings before respecting headings would discard author intent. Very short sections should remain intact.

## Chroma metadata

Each chunk has source=notion, notion_page_id, title, section, url, block_ids, last_edited_time, page_content_hash, chunk_content_hash, chunk_index, and scalar prop_* values for supported page properties.

The script scopes stale-vector cleanup to source=notion and one notion_page_id; it never clears the whole collection. Existing sources and demo documents remain intact.

The app.tools update also fixes a shared-collection edge case: initialization now inserts missing demo IDs even when Notion documents have already populated the collection. Retrieval now includes source/title/section/URL text in each returned document, so the existing finalizer can cite the Notion page without changing the LangGraph graph topology.

## PostgreSQL for structured application data

Recommended database: PostgreSQL 18, the current stable major release as of October 2026. It is permissively open source under the PostgreSQL License. Official project: https://www.postgresql.org/ and license: https://www.postgresql.org/about/licence/.

Brief comparison:
- SQLite: easiest embedded/local demo, no server, but limited concurrent write workflows and operational features.
- PostgreSQL: durable relational database, constraints, transactions, indexing, JSONB, concurrent access, and first-class Docker support. Best fit for structured project data and operational state.
- MySQL/MariaDB: also capable relational databases; PostgreSQL is the simpler recommendation here because it fits the use case and the repository has no existing MySQL dependency.

The Compose service is in compose.postgres.yaml. Start it:

    docker compose -f compose.postgres.yaml up -d
    docker compose -f compose.postgres.yaml ps

Stop it while retaining the persistent volume:

    docker compose -f compose.postgres.yaml down

Only use docker compose down -v when intentionally deleting all PostgreSQL data. Change the sample password before using this outside a local development machine.

Connect from Python using Psycopg 3:

    import psycopg

    with psycopg.connect("postgresql://agentic_rag:local_dev_change_me@localhost:5432/agentic_rag") as conn:
        row = conn.execute("SELECT version()").fetchone()
        print(row[0])

The helper app.postgres_store creates two tables:
- ingestion_sync: source, page/item identifier, last content hash, last edit time, chunk count, sync status, and sync timestamp.
- agent_runs: run ID, original/rewritten query, route, status, timestamps, and compact JSONB details.

Initialize the schema and record a checkpoint:

    from app.postgres_store import initialize_schema, record_ingestion_sync

    initialize_schema()
    record_ingestion_sync(
        source="notion",
        item_id="NOTION_PAGE_UUID",
        content_hash="sha256-of-normalized-page",
        chunk_count=6,
        last_edited_time="2026-10-30T16:35:16.514Z",
    )

The helper is optional; configure DATABASE_URL only when PostgreSQL is running. The Notion sync script automatically writes a page-level checkpoint to ingestion_sync after successful Chroma upserts when DATABASE_URL is set; a checkpoint failure is logged without undoing the Chroma write. The existing generated-SQL agent branch still uses the read-only/write-gated SQLite demo database in data/demo.db. This incremental change does not silently replace that execution path. When you begin storing business records in PostgreSQL, connect that branch through a separate, explicitly configured SQL adapter and preserve its approval/read-only protections.

## Architecture boundary

- Notion: authoring/source of truth for prose knowledge, project notes, and editable status/date properties.
- ChromaDB: embeddings of semantically useful prose plus source/citation/filter metadata.
- PostgreSQL: normalized relational application data, transactional tasks/records when needed, source sync checkpoints, and agent-run metadata.
- Existing LangGraph: unchanged orchestration. Its RAG subgraph searches the shared Chroma collection, judges relevance, refines within the configured retry limit, and uses the existing web fallback when evidence is insufficient.

For task status and due-date questions, query structured Notion/SQL properties; do not answer from vector similarity. For architecture and explanation questions, retrieve prose chunks from Chroma.
