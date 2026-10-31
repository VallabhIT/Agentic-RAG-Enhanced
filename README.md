# Agentic RAG — Reliability, Evaluation & Cost-Aware Tool Use

A small LangGraph assistant that demonstrates practical agent engineering: bounded routing, hybrid RAG, guarded SQLite/Python tools, safe failure handling, and repeatable evaluation.

The design intentionally stays simple:
- **LangGraph** for orchestration and explicit state transitions.
- **ChromaDB** for semantic + lexical hybrid retrieval over Notion/local knowledge.
- **SQLite** for structured Technology Department client-deliverable records.
- **Notion** for personal/project knowledge pages.
- **Web/Python** only when the request needs those capabilities.

## Current Notion knowledge sources

The current workspace structure under **My Personal Things** contains separate pages for:
- Agentic RAG — Project Notes & Integration Plan
- Himachal Pradesh Winter Itinerary | 1–8 Jan 2027
- Grocery Shopping List
- Important Events in Indian History
- Weekday routine

The ingestion script handles headings, paragraphs, lists, code, callouts, task items, and Notion tables. Each page is supplied explicitly through `NOTION_PAGE_IDS`; the parent page is only a navigation page and does not need to be indexed. Do not put a client-deliverables database into the Notion ingestion list.

## Agent workflow

```
new turn -> rewrite -> route
                     |-- rag -> hybrid retrieve -> judge -> (bounded refine/retrieve)* -> finalize
                     |                   -> insufficient -> web fallback -> finalize
                     |-- sql -> generate SQL -> safety guard -> execute -> finalize
                     |                           -> approval required -> pause/resume
                     |-- python -> generate -> safety guard -> execute -> finalize
                     |                              -> approval required -> pause/resume
                     |-- web -> web search -> finalize
                     -- direct -> conversational response
```

## Hybrid retrieval

RAG retrieval uses two independent signals:

1. **Dense Chroma retrieval** for semantic paraphrases and conceptual similarity.
2. **Corpus-wide BM25 lexical retrieval** for exact entities, names, years, numbers, quantities, units, and identifiers.
3. **Weighted fusion** ranks the union of both result sets.

This matters for pages such as the grocery list and Indian history timeline, where queries containing `5 kg`, `2 L`, `1947`, or exact item/event names can be underweighted by embeddings alone.

Tune with:

```text
RAG_TOP_K
RAG_HYBRID_CANDIDATES
RAG_HYBRID_VECTOR_WEIGHT
RAG_HYBRID_LEXICAL_WEIGHT
RAG_HYBRID_BM25_K1
RAG_HYBRID_BM25_B
```

## Routing policy

- Agentic RAG concepts, prompts, failure modes, evaluation, security, observability, and optimization -> RAG.
- Personal/project knowledge from the indexed Notion pages -> RAG.
- Client deliverables, owners, status, blockers, progress, and due dates -> SQLite.
- Standalone computations/scripts -> Python.
- Current public information -> Web.
- Greetings/thanks -> Direct.

The router explicitly distinguishes conceptual questions from structured tracker requests, so a question like "What does blocked mean?" stays in RAG while "Show blocked deliverables" uses SQLite.

## SQLite client-deliverables database

The project now owns a direct local database at `data/client_deliverables.sqlite`. Run:

```bash
python -m scripts.init_sqlite
```

The schema is `client_deliverables` with client, project, deliverable, workstream, owner, status, priority, due date, progress percentage, blocker, next action, last update, and `is_sample`.

Seed rows are fictional illustrative records only. Replace them before using the application for real reporting. No PostgreSQL, external SQL server, Docker database, or Notion deliverables database is required.

## Evaluation suite

The repository separates evaluation by scope:

### Routing

```bash
python -m scripts.evaluate_routing
```

Measures route accuracy, confusion counts, latency percentiles, invalid route outputs, and deterministic-router call avoidance.

### Retrieval

```bash
python -m scripts.evaluate_retrieval
```

Measures source-level Recall@k and MRR over `evals/retrieval_cases.json`.

### RAG subgraph / DeepEval

```bash
python -m scripts.evaluate_rag
```

The RAG suite uses five complementary metrics:
- **Contextual Precision** — whether relevant chunks are ranked early.
- **Contextual Recall** — whether retrieved chunks contain the evidence needed by the reference answer.
- **Contextual Relevancy** — how much of the retrieved context is relevant to the query.
- **Faithfulness** — whether the generated answer is supported by retrieved context.
- **Answer Relevancy** — whether the answer addresses the user's question.

### Node-level checks

```bash
python -m scripts.evaluate_nodes
```

Covers SQL safety, Python safety, bounded RAG retry invariants, and rewrite preservation.

### Application-level

```bash
python -m scripts.evaluate_pipeline
```

Runs the golden end-to-end cases and reports route accuracy, DeepEval Answer Relevancy, and Tool Correctness. Trajectory metrics such as Task Completion and Step Efficiency are intentionally deferred until runtime tracing is added; the project does not pretend to have a trajectory when it does not capture one.

The goldens live under `evals/` and should be versioned with the application so prompt/retrieval changes can be regression-tested.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY (or configure LLM_PROVIDER=ollama)
python -m scripts.init_sqlite
python -m app.main
```

## Notion ingestion

Create a local Notion integration and share only the pages intended for ingestion. Set `NOTION_TOKEN` and the comma-separated `NOTION_PAGE_IDS` in `.env`.

```bash
python -m scripts.notion_to_chroma
```

The current `.env.example` lists the five knowledge pages from the workspace. The parent **My Personal Things** page is intentionally not indexed because it is only a navigation page. Re-run the sync after a Notion edit; unchanged pages are skipped using content hashes.

## Example questions

- "Why does hybrid retrieval help with exact numbers?" (RAG)
- "How much rice and cooking oil are on my grocery list?" (RAG)
- "What happened in 1947 in my history timeline?" (RAG)
- "What is planned for Day 4 of my Himachal itinerary?" (RAG)
- "What is in my weekday routine at 8:30 AM?" (RAG)
- "Show blocked deliverables." (SQLite)
- "Compute precision, recall, and F1 for TP=42, FP=8, FN=5." (Python)
- "What is the latest public LangGraph release?" (Web)
