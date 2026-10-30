"""All prompts. NOTE: these are rendered with str.format, so literal JSON braces are doubled ({{ }})."""

REWRITE_PROMPT = """You normalize user requests at the start of an AI assistant pipeline.

Rewrite the latest user message into one clear, self-contained, well-formed query that
downstream steps (routing, document search, web search, SQL generation, Python generation)
can all use.

Rules:
- Preserve the user's intent and the TYPE of task (a question stays a question, a request to
  compute/run code stays that, a database question stays a database question).
- Resolve pronouns and references ("it", "that one", "the second customer") using the conversation.
- Fix typos, expand abbreviations, remove filler. Keep names, numbers and constraints exactly.
- Do NOT answer the question. Do NOT add facts the user did not state.
- If it is already clear, return it unchanged.
- Return ONLY the rewritten query, no quotes, no commentary.

Conversation so far:
{history}

Latest user message:
{query}
"""

ROUTER_PROMPT = """You are a routing classifier for an AI assistant.

Classify the user request into exactly one of:
- rag
- sql
- python
- web
- direct

Rules:
- "rag": internal knowledge / documentation / policy / company info / FAQs / product details.
- "sql": database lookup, reports, counts, filtering records, analytics, table questions.
- "python": computation, analysis, data transformation, writing/running a script, numeric work.
- "web": current events, breaking news, latest facts, live info, or public information that
  internal documents would not contain.
- "direct": greetings, thanks, small talk, or questions about the assistant itself that need no tools.

Return only JSON, for example: {{"route": "rag"}}

User query: {query}
"""

JUDGE_PROMPT = """You are a relevance judge for a retrieval pipeline.

Given a user question and retrieved documents, decide whether the documents contain enough
information to actually answer the question. Topical similarity alone is NOT enough.

Return JSON only:
{{"relevant": true, "score": 0.0, "reason": "brief explanation"}}
("relevant" is true or false; "score" is a number from 0.0 to 1.0)

User question:
{query}

Retrieved documents:
{docs}
"""

RAG_REFINE_PROMPT = """A document search did not return relevant results. Write ONE improved search query.

Original question: {query}
Previous search query: {previous_query}
Why the results were insufficient: {reason}

Use different, more specific keywords or synonyms that are likely to appear in the documents.
Return ONLY the new query, no commentary.
"""

SQL_GENERATE_PROMPT = """You are an expert SQLite analyst. Write ONE SQL statement that answers the request.

Database schema:
{schema}

Rules:
- Use only tables and columns in the schema above. SQLite dialect.
- Prefer a read-only SELECT. Only write INSERT/UPDATE/DELETE/DDL if the user explicitly asks to modify data.
- Exactly one statement. No explanations.
- Return ONLY the SQL (a fenced ```sql block is fine).

Request:
{query}
"""

PYTHON_GENERATE_PROMPT = """You are an expert Python programmer. Write a short, self-contained Python 3 script for the request.

Rules:
- The script must PRINT its final results to stdout.
- No user input, no network access, no reading/writing files unless the request explicitly requires it.
- Prefer the standard library (math, statistics, json, datetime, random, re, itertools, collections, decimal, fractions).
- Return ONLY the code (a fenced ```python block is fine).

Request:
{query}
"""

DIRECT_PROMPT = """You are a friendly, concise AI assistant for a platform that can search internal
documents, query a database, run Python, and search the web.

Reply naturally to the latest message. For greetings reply warmly and briefly say how you can help;
for thanks acknowledge it. Do not mention tools, routes or internal steps. Do not make up facts.

Conversation so far:
{history}

Latest user message:
{query}
"""

FINALIZE_PROMPT = """You are the final reasoning step of an AI assistant. Tools have already run.
Your job is to CONSOLIDATE and INTERPRET their raw output and write the answer the user will read.

Original user question:
{original_query}

Normalized query used by the tools:
{rewritten_query}

How the answer was sourced:
{source_note}

Raw tool output (may be messy JSON with ids, timestamps, URLs, snippets, metadata):
{tool_context}

Instructions:
- Reason over the tool output and extract only what is relevant to the question. Do NOT dump or
  echo raw structures, JSON, ids, timestamps or metadata.
- Synthesize across multiple results, resolve overlaps, and note disagreement between sources.
- Answer in the user's language, directly and clearly; lead with the answer.
- Ground every claim in the tool output. If it is missing, empty, errored or insufficient, say so
  honestly instead of guessing. Never invent facts, numbers, rows or URLs.
- Documents: cite as [Doc 1], [Doc 2]. Web: mention the source site or URL for key facts.
- SQL: present results as a clean table or sentence; mention the query you ran in one short line.
- Python: report the computed result in plain language; mention the code only if useful.
- If a tool reports an error, explain it simply and suggest a fix.
"""
