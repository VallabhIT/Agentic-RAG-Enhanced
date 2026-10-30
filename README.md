# Agentic LangGraph RAG Project

A LangGraph assistant that rewrites the query, routes it, and uses RAG, web search, SQL and Python tools
(with a human-approval gate for risky code), then consolidates tool output into a clean final answer.

## Flow

```
START ─ entry_gate ─┬─ new turn ──► rewrite ──► router ─┬─ rag ─► relevant? ─ yes ───────────────► finalize
                    │                 │                 │             └ no (after retries) ─► web_search ─► finalize
                    │                 └ greeting/thanks ┼─ web ────► web_search ─► finalize
                    │                   (skip tools)    ├─ sql ────► sql_generate ─► sql_guard ─┬ safe ─► sql_execute ─► finalize
                    │                                   │                                       └ risky ─► pause for approval
                    │                                   └─ python ─► python_generate ─► python_guard ─ (same pattern)
                    ├─ approval granted ─► sql_execute / python_execute   (no rewrite, no routing, no re-check)
                    └─ approval denied  ─► finalize
```

Key behaviours

- **Query rewriting first.** `original_query` is preserved; `rewritten_query` is what routing, RAG judging/retrieval,
  web search and SQL/Python generation use. The finalize node sees both.
- **Finalize = reasoning, not summarising.** It receives the *full raw* tool output (web-result dicts, doc chunks,
  SQL rows, script stdout) and writes a clean answer. Greetings/thanks are answered conversationally and never touch tools.
- **Generated code is stored and executed.** The LLM writes the SQL / Python (`sql_query`, `python_code` in state);
  the execute node runs that code, never the natural-language question.
- **Explicit approval state.** `approval_status` (`not_required | pending | granted | denied`), `pending_action`,
  and `pending_payload` (the exact code shown to the human). After approval the graph re-enters at the execute node.
  Execution also refuses if the code differs from what was approved.
- **RAG → web fallback on actual relevance.** retrieve → LLM judge → (refine query → retrieve) up to `RAG_MAX_RETRIES`.
  If the final chunks are still not relevant, the web tool runs; otherwise the RAG docs go to finalize.
- **Safety.** Only single read-only `SELECT/WITH` runs without approval (and on a read-only SQLite connection).
  Python runs in a separate interpreter with a timeout and a scrubbed environment; imports outside a small
  stdlib whitelist, `open/eval/exec`, dunder access etc. need approval.
  This is process isolation, not a hardened sandbox - use a container for untrusted production use.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your OPENAI_API_KEY (or set LLM_PROVIDER=ollama)
python -m app.main
```

## Tests (offline, no API key or network)

```bash
python -m pytest tests -q
```

## Example queries

- `hi` / `thanks` (conversational, no tools)
- What does the product do? (RAG)
- What is the latest news on OpenAI? (web)
- Which customers spent more than 2000? (generated SQL)
- Delete customer with id 1 (generated SQL, asks for approval once)
- Compute the first 10 Fibonacci numbers (generated Python)
