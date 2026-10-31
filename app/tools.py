import ast
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    DATA_DIR,
    PYTHON_TIMEOUT_SECONDS,
    RAG_HYBRID_BM25_B,
    RAG_HYBRID_BM25_K1,
    RAG_HYBRID_CANDIDATES,
    RAG_HYBRID_LEXICAL_WEIGHT,
    RAG_HYBRID_VECTOR_WEIGHT,
    SQL_MAX_ROWS,
    SQLITE_DB_PATH,
    VECTORSTORE_PATH,
)

# --------------------------------------------------------------------------- #
# Vector store / hybrid RAG retrieval
# --------------------------------------------------------------------------- #

def load_demo_documents():
    from langchain_core.documents import Document

    file_path = DATA_DIR / "demo_docs.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=item["content"],
            metadata={
                "title": item["title"], "id": item["id"], "source": "demo",
                "content_hash": __import__("hashlib").sha256(item["content"].encode("utf-8")).hexdigest(),
            },
        )
        for item in data
    ]


@lru_cache(maxsize=1)
def _get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_vectorstore():
    from langchain_chroma import Chroma
    return Chroma(
        collection_name="knowledge_base",
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=_get_embeddings(),
    )


def init_vectorstore(force_rebuild: bool = False):
    vectorstore = _get_vectorstore()
    if force_rebuild:
        vectorstore.delete_collection()
        _get_vectorstore.cache_clear()
        vectorstore = _get_vectorstore()
    docs = load_demo_documents()
    demo_ids = [str(d.metadata["id"]) for d in docs]
    existing = vectorstore.get(ids=demo_ids, include=["metadatas"])
    existing_meta = dict(zip(existing.get("ids") or [], existing.get("metadatas") or []))
    changed = [
        doc for doc in docs
        if existing_meta.get(str(doc.metadata["id"]), {}).get("content_hash") != doc.metadata["content_hash"]
    ]
    if changed:
        vectorstore.add_documents(changed, ids=[str(d.metadata["id"]) for d in changed])
    return vectorstore


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", (text or "").lower())


def _bm25_scores(query: str, documents: list[Any]) -> list[float]:
    """Small dependency-free BM25 scorer over the retrieved Chroma candidate set."""
    q = _tokens(query)
    if not q or not documents:
        return [0.0] * len(documents)
    tokenized = [_tokens(getattr(d, "page_content", "")) for d in documents]
    lengths = [len(x) for x in tokenized]
    avgdl = sum(lengths) / max(len(lengths), 1)
    df: dict[str, int] = {}
    for toks in tokenized:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    n = len(tokenized)
    scores = []
    for toks, dl in zip(tokenized, lengths):
        tf = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1.0 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            freq = tf[term]
            denom = freq + RAG_HYBRID_BM25_K1 * (1 - RAG_HYBRID_BM25_B + RAG_HYBRID_BM25_B * dl / max(avgdl, 1.0))
            score += idf * (freq * (RAG_HYBRID_BM25_K1 + 1)) / max(denom, 1e-9)
        scores.append(score)
    return scores


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_docs(query: str, k: int = 5) -> List[str]:
    """Hybrid retrieval: dense Chroma candidates + BM25 lexical scoring + weighted fusion.

    Dense retrieval handles semantic paraphrases; lexical scoring preserves exact entities,
    names, years, quantities, units, and identifiers that embeddings may underweight.
    """
    vectorstore = init_vectorstore()
    candidate_k = max(k, RAG_HYBRID_CANDIDATES)
    candidates = vectorstore.similarity_search_with_score(query, k=candidate_k)
    if not candidates:
        return []

    docs = [doc for doc, _ in candidates]
    distances = [float(score) for _, score in candidates]
    dense_similarity = [1.0 / (1.0 + max(d, 0.0)) for d in distances]
    lexical = _bm25_scores(query, docs)
    dense_norm = _minmax(dense_similarity)
    lexical_norm = _minmax(lexical)
    fused = [
        RAG_HYBRID_VECTOR_WEIGHT * d + RAG_HYBRID_LEXICAL_WEIGHT * b
        for d, b in zip(dense_norm, lexical_norm)
    ]
    ranked = sorted(zip(fused, docs), key=lambda x: x[0], reverse=True)[:k]

    formatted = []
    for _, doc in ranked:
        metadata = doc.metadata or {}
        source = metadata.get("source", "local")
        title = metadata.get("title", "")
        section = metadata.get("section", "")
        url = metadata.get("url", "")
        citation = " | ".join(str(x) for x in (source, title, section, url) if x)
        formatted.append((f"Source: {citation}\n\n" if citation else "") + doc.page_content)
    return formatted


# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return [dict(r) for r in DDGS().text(query, max_results=max_results)]
    except Exception as exc:
        return [{"error": f"Web search failed: {exc}"}]


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #

_WRITE_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|ALTER|INSERT|REPLACE|CREATE|TRUNCATE|EXEC|EXECUTE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|LOAD_EXTENSION)\b",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def is_dangerous_sql(sql: str) -> bool:
    cleaned = _strip_sql_comments(sql or "")
    if not cleaned:
        return True
    first = cleaned.lstrip("(").split(None, 1)[0].upper()
    return first not in {"SELECT", "WITH"} or bool(_WRITE_KEYWORDS.search(cleaned))


def ensure_demo_db() -> Path:
    """Create the direct SQLite client-deliverables database if it is missing."""
    db_path = SQLITE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS client_deliverables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client TEXT NOT NULL,
                project TEXT NOT NULL,
                deliverable TEXT NOT NULL,
                workstream TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Not Started','In Progress','Blocked','In Review','Delivered')),
                priority TEXT NOT NULL CHECK(priority IN ('Critical','High','Medium','Low')),
                due_date TEXT,
                progress_pct INTEGER NOT NULL DEFAULT 0 CHECK(progress_pct BETWEEN 0 AND 100),
                blocker TEXT NOT NULL DEFAULT '',
                next_action TEXT NOT NULL DEFAULT '',
                last_update TEXT,
                is_sample INTEGER NOT NULL DEFAULT 1 CHECK(is_sample IN (0,1)),
                UNIQUE(client, project, deliverable)
            )
        """)
        count = conn.execute("SELECT COUNT(*) FROM client_deliverables").fetchone()[0]
        if count == 0:
            seed_file = DATA_DIR / "client_deliverables.json"
            if seed_file.exists():
                rows = json.loads(seed_file.read_text(encoding="utf-8"))
                conn.executemany("""
                    INSERT OR IGNORE INTO client_deliverables
                    (client, project, deliverable, workstream, owner, status, priority, due_date,
                     progress_pct, blocker, next_action, last_update, is_sample)
                    VALUES (:client,:project,:deliverable,:workstream,:owner,:status,:priority,:due_date,
                            :progress_pct,:blocker,:next_action,:last_update,1)
                """, rows)
        conn.commit()
    finally:
        conn.close()
    return db_path


def get_schema() -> str:
    db_path = ensure_demo_db()
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='client_deliverables'").fetchone()
    finally:
        conn.close()
    return ("-- Seeded starter rows are illustrative only (is_sample=1), not verified client commitments.\n" + row[0]) if row else "(client_deliverables table not available)"


def route_hint(query: str) -> Optional[str]:
    q = (query or "").lower().strip()
    if not q:
        return None
    if re.match(r"^(explain|define|describe|what does .+ mean|what is (?:a|an) .+|how does .+ work)\b", q):
        return None
    entity_terms = ("deliverable", "deliverables", "client", "clients", "project", "projects", "workstream", "owner", "owners", "deadline", "due date", "blocker", "blockers")
    state_terms = ("status", "blocked", "overdue", "due", "progress", "priority", "assigned", "delivered", "in review", "in progress", "next action")
    report_actions = ("show", "list", "find", "which", "who", "how many", "count", "report", "summarize", "display", "compare", "filter", "sort", "track", "give me")
    if any(x in q for x in entity_terms) and (any(x in q for x in state_terms) or any(x in q for x in report_actions)):
        return "sql"
    if any(x in q for x in ("what is due", "what's due", "who is assigned", "who owns", "which deliverables", "number of deliverables", "delivery status", "client status", "project status")):
        return "sql"
    if any(x in q for x in ("latest news", "breaking news", "current events", "today's news", "right now", "current price", "latest release", "latest version", "recent announcement", "search the web", "look up online")):
        return "web"
    if any(x in q for x in ("calculate", "compute", "run python", "write a python", "write python", "create a script", "execute python", "plot ")):
        return "python"
    return None


def run_sql(query: str) -> dict:
    db_path = ensure_demo_db()
    read_only = not is_dangerous_sql(query)
    conn = sqlite3.connect(db_path.as_uri() + ("?mode=ro" if read_only else "?mode=rw"), uri=True)
    try:
        cur = conn.execute(query)
        if cur.description:
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchmany(SQL_MAX_ROWS + 1)]
            return {"columns": columns, "rows": rows[:SQL_MAX_ROWS], "row_count": min(len(rows), SQL_MAX_ROWS), "truncated": len(rows) > SQL_MAX_ROWS}
        conn.commit()
        return {"columns": [], "rows": [], "row_count": 0, "rows_affected": cur.rowcount, "truncated": False}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

SAFE_IMPORTS = {"math", "statistics", "json", "datetime", "random", "re", "itertools", "collections", "decimal", "fractions", "string", "functools", "operator", "heapq", "bisect", "textwrap"}
_BLOCKED_CALLS = {"exec", "eval", "compile", "open", "__import__", "input", "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars", "exit", "quit"}
_BLOCKED_ATTRS = {"system", "popen", "check_output", "check_call", "call", "run", "Popen", "spawn", "remove", "rmdir", "unlink", "rmtree", "exec", "eval"}


def detect_dangerous_python(code: str) -> bool:
    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] not in SAFE_IMPORTS for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] not in SAFE_IMPORTS):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_ATTRS:
                return True
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True
        if isinstance(node, ast.Name) and node.id in {"__builtins__", "__import__"}:
            return True
    return False


def run_python(code: str, timeout: Optional[int] = None) -> dict:
    timeout = timeout or PYTHON_TIMEOUT_SECONDS
    env = {"PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
    if os.name == "nt" and os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1, "timed_out": True}
    return {"stdout": proc.stdout[-8000:], "stderr": proc.stderr[-4000:], "returncode": proc.returncode, "timed_out": False}
