# Agentic RAG — Reliability, Evaluation & Cost-Aware Tool Use

A small LangGraph assistant that demonstrates practical agent engineering: bounded routing, evidence-based RAG, guarded SQLite/Python tools, safe failure handling, and an offline evaluation set.

The design intentionally stays simple:
- **LangGraph** for orchestration and explicit state transitions.
- **ChromaDB** for semantic search over agentic-RAG engineering knowledge.
- **SQLite** for structured technology-department client deliverables.
- **Notion** for authoring explanatory knowledge and managing the separate client-deliverables tracker.
- **Web/Python** only when the request needs those capabilities.

## Agent workflow

```
new turn -> rewrite -> route
                     |-- rag -> retrieve -> judge -> (bounded refine/retrieve)* -> finalize
                     |                   -> insufficient -> web fallback -> finalize
                     |-- sql -> generate SQL -> safety guard -> execute -> finalize
                     |                           -> approval required -> pause/resume
                     |-- python -> generate -> safety guard -> execute -> finalize
                     |                              -> approval required -> pause/resume
                     |-- web -> web search -> finalize
                     -- direct -> conversational response
```

### Engineering decisions worth explaining in an interview

**1. Failure modes are explicit**

- Invalid router JSON safely falls back to internal RAG rather than crashing.
- RAG judges answer sufficiency, not topic similarity; retries are bounded by `RAG_MAX_RETRIES`.
- Insufficient internal evidence triggers the existing web fallback.
- SQL is executed as one statement; read-only `SELECT/WITH` uses a read-only SQLite connection.
- Risky SQL and Python are paused for explicit approval. The execute node checks that the code still matches the exact approved payload.
- Python runs in a separate subprocess with a timeout and scrubbed environment. This is process isolation, not a hardened sandbox.
- Tool exceptions are surfaced as typed status/error information instead of silently inventing an answer.

**2. Routing is grounded in the actual sources**

Obvious intent hints avoid a router LLM call for simple tracker reports, public freshness requests, and explicit standalone computations. Ambiguous queries go to the LLM router, which receives the live SQLite schema and a source-specific routing policy.

- Agentic RAG concepts, prompts, failure modes, evaluation, security, and optimization -> RAG.
- Client deliverables, clients, workstreams, owners, status, blockers, progress, and due dates -> SQLite.
- Standalone computations/scripts -> Python.
- Current public information -> Web.
- Greetings/thanks -> Direct.

The SQL generator receives only the live `client_deliverables` schema. It is not asked to invent customer/order tables.

**3. Evaluation is part of the workflow**

`evals/routing_cases.json` is a small, versioned golden set. Offline tests cover deterministic route hints, ambiguous-vs-structured distinctions, prompt schema context, RAG retries/fallback, approval resume, SQL safety, and Python isolation.

For a stronger evaluation report, add labeled expected evidence to the set and measure:
- Route accuracy and per-route precision/recall/confusion matrix.
- Retrieval Recall@k and MRR.
- Answer faithfulness and citation coverage.
- SQL execution correctness and safety-guard false positives/negatives.
- Tool success rate, retry rate, and fallback correctness.

Do not report a model quality score from the deterministic hint tests; those tests cover only the inexpensive first routing layer. Evaluate the LLM router and final answers separately against the same golden queries when credentials are configured.

**4. Optimisation follows measurement**

- Cache the embedding model and Chroma client.
- Upsert only changed seed knowledge by content hash.
- Keep top-k and context budgets bounded.
- Avoid repeated LLM calls for obvious intents.
- Use a small retry budget and stop when evidence is sufficient.
- Prefer structural chunking and add reranking/semantic chunking only when evals demonstrate a retrieval failure.
- Track p50/p95 latency and token usage per node before adding complexity.

## Knowledge and structured data

### ChromaDB
Stores explanatory material such as agentic workflow design, RAG retrieval, routing policy, tool safety, failure recovery, evaluation, observability, and cost optimization. Notion page title/section/URL provenance is carried into retrieved context.

### SQLite
Stores structured client-deliverable records in `client_deliverables`:
- client, project, deliverable, workstream
- owner, status, priority
- due date, progress percentage
- blocker/dependency, next action, last update
- `is_sample`, which marks the seeded example records

The initial rows are illustrative sample data, not actual client commitments. Replace them before using this as a real status report. No PostgreSQL, external SQL server, or Docker database service is required.

### Notion
Use Notion for explanatory knowledge pages and the **Technology Department — Client Deliverables Tracker**. The tracker is operational data and should be queried as structured rows, not embedded as text in Chroma.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY (or configure LLM_PROVIDER=ollama)
python -m app.main
```

The SQLite deliverables table and illustrative rows are created automatically the first time the SQL schema is requested. The existing database path remains `data/demo.db`.

## Notion ingestion

Create a local Notion integration, share the intended knowledge pages with it, and set `NOTION_TOKEN` and `NOTION_PAGE_IDS` in `.env`.

```bash
python -m scripts.notion_to_chroma --page-ids YOUR_NOTION_PAGE_ID
```

This writes explanatory page chunks into the existing Chroma `knowledge_base` collection. Do not pass the client-deliverables database rows as RAG documents.

## Evaluation and offline tests

Run the live routing evaluation after configuring the existing LLM provider. The report includes route accuracy, per-route confusion counts, latency percentiles, invalid outputs, and the fraction of calls avoided by deterministic hints:

```bash
python -m scripts.evaluate_routing
```

This evaluates routing only; it does not claim to measure retrieval quality or answer faithfulness. The offline tests cover deterministic hints and the graph's retry, fallback, approval, SQL safety, and Python isolation behavior:

```bash
python -m pytest tests -q
```

## Example questions

- "How should an agent recover from irrelevant RAG results?" (RAG)
- "How do I evaluate retrieval faithfulness and citation coverage?" (RAG)
- "Show blocked deliverables for Northstar Logistics." (SQLite)
- "Which deliverables are due this week?" (SQLite)
- "Compute precision, recall, and F1 for TP=42, FP=8, FN=5." (Python)
- "What is the latest public LangGraph release?" (Web)
