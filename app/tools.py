import ast
import json
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
    SQL_MAX_ROWS,
    SQLITE_DB_PATH,
    VECTORSTORE_PATH,
)

# --------------------------------------------------------------------------- #
# Vector store / RAG
# --------------------------------------------------------------------------- #

def load_demo_documents():
    from langchain_core.documents import Document

    file_path = DATA_DIR / "demo_docs.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    import hashlib

    return [
        Document(
            page_content=item["content"],
            metadata={
                "title": item["title"],
                "id": item["id"],
                "source": "demo",
                "content_hash": hashlib.sha256(item["content"].encode("utf-8")).hexdigest(),
            },
        )
        for item in data
    ]


@lru_cache(maxsize=1)
def _get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older installs
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
    """Ensure demo documents exist without overwriting other sources in the shared collection."""
    vectorstore = _get_vectorstore()

    if force_rebuild:
        vectorstore.delete_collection()
        _get_vectorstore.cache_clear()
        vectorstore = _get_vectorstore()

    # Upsert only missing or changed demo documents. This also refreshes seed text
    # after a project knowledge-base update without rebuilding other source vectors.
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


def retrieve_docs(query: str, k: int = 5) -> List[str]:
    """Return retrieved text with source provenance while retaining the current string contract."""
    vectorstore = init_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    formatted = []
    for doc in results:
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
    """Return structured DuckDuckGo results (title / href / body). Never raises."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        return [dict(r) for r in results]
    except Exception as exc:
        return [{"error": f"Web search failed: {exc}"}]


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #

_WRITE_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|ALTER|INSERT|REPLACE|CREATE|TRUNCATE|EXEC|EXECUTE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|LOAD_EXTENSION)\b",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def is_dangerous_sql(sql: str) -> bool:
    """True if the statement is anything other than a plain read-only SELECT/WITH query."""
    cleaned = _strip_sql_comments(sql or "")
    if not cleaned:
        return True
    first = cleaned.lstrip("(").split(None, 1)[0].upper()
    if first not in {"SELECT", "WITH"}:
        return True
    return bool(_WRITE_KEYWORDS.search(cleaned))


def ensure_demo_db() -> Path:
    """Create the simple SQLite client-deliverables table and seed illustrative records once."""
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
                status TEXT NOT NULL CHECK(status IN
                    ('Not Started', 'In Progress', 'Blocked', 'In Review', 'Delivered')),
                priority TEXT NOT NULL CHECK(priority IN ('Critical', 'High', 'Medium', 'Low')),
                due_date TEXT,
                progress_pct INTEGER NOT NULL DEFAULT 0 CHECK(progress_pct BETWEEN 0 AND 100),
                blocker TEXT NOT NULL DEFAULT '',
                next_action TEXT NOT NULL DEFAULT '',
                last_update TEXT,
                is_sample INTEGER NOT NULL DEFAULT 1 CHECK(is_sample IN (0, 1)),
                UNIQUE(client, project, deliverable)
            )
        """)
        count = conn.execute("SELECT COUNT(*) FROM client_deliverables").fetchone()[0]
        if count == 0:
            seed_file = DATA_DIR / "client_deliverables.json"
            if seed_file.exists():
                seed_rows = json.loads(seed_file.read_text(encoding="utf-8"))
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO client_deliverables
                    (client, project, deliverable, workstream, owner, status, priority,
                     due_date, progress_pct, blocker, next_action, last_update, is_sample)
                    VALUES
                    (:client, :project, :deliverable, :workstream, :owner, :status, :priority,
                     :due_date, :progress_pct, :blocker, :next_action, :last_update, 1)
                    """,
                    seed_rows,
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


def get_schema() -> str:
    """Expose only the real client-deliverables table to the SQL generator."""
    db_path = ensure_demo_db()
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='client_deliverables'"
        ).fetchone()
    finally:
        conn.close()
    return ("-- Seeded starter rows are illustrative only (is_sample=1), not verified client commitments.\n" + row[0]) if row else "(client_deliverables table not available)"


def route_hint(query: str) -> Optional[str]:
    """Deterministically route obvious requests; leave ambiguous intent to the LLM router."""
    q = (query or "").lower().strip()
    if not q:
        return None

    # Definition/explanation phrasing is knowledge work, even when it mentions
    # tracker vocabulary such as "blocked", "priority", or "deliverable".
    if re.match(r"^(explain|define|describe|what does .+ mean|what is (?:a|an) .+|how does .+ work)\b", q):
        return None

    entity_terms = (
        "deliverable", "deliverables", "client", "clients", "project", "projects",
        "workstream", "owner", "owners", "deadline", "due date", "blocker", "blockers",
    )
    state_terms = (
        "status", "blocked", "overdue", "due", "progress", "priority", "assigned",
        "delivered", "in review", "in progress", "next action",
    )
    report_actions = (
        "show", "list", "find", "which", "who", "how many", "count", "report",
        "summarize", "display", "compare", "filter", "sort", "track", "give me",
    )
    has_entity = any(term in q for term in entity_terms)
    has_state = any(term in q for term in state_terms)
    has_action = any(term in q for term in report_actions)

    # Structured delivery questions take precedence over "latest/current" wording.
    if has_entity and (has_state or has_action):
        return "sql"
    if any(phrase in q for phrase in (
        "what is due", "what's due", "who is assigned", "who owns",
        "which deliverables", "number of deliverables", "delivery status",
        "client status", "project status",
    )):
        return "sql"

    # Public freshness cues only apply after structured tracker intent has been checked.
    if any(phrase in q for phrase in (
        "latest news", "breaking news", "current events", "today's news",
        "right now", "current price", "latest release", "latest version",
        "recent announcement", "search the web", "look up online",
    )):
        return "web"

    if any(phrase in q for phrase in (
        "calculate", "compute", "run python", "write a python", "write python",
        "create a script", "execute python", "plot ", "calculate the",
    )):
        return "python"

    return None


def run_sql(query: str) -> dict:
    """Execute exactly one statement; unapproved read queries use SQLite read-only mode."""
    db_path = ensure_demo_db()
    read_only = not is_dangerous_sql(query)
    uri = db_path.as_uri() + ("?mode=ro" if read_only else "?mode=rw")
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(query)  # sqlite3 rejects multiple statements in one execute call
        if cur.description:
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchmany(SQL_MAX_ROWS + 1)]
            truncated = len(rows) > SQL_MAX_ROWS
            return {
                "columns": columns,
                "rows": rows[:SQL_MAX_ROWS],
                "row_count": min(len(rows), SQL_MAX_ROWS),
                "truncated": truncated,
            }
        conn.commit()
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "rows_affected": cur.rowcount,
            "truncated": False,
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

SAFE_IMPORTS = {
    "math", "statistics", "json", "datetime", "random", "re", "itertools", "collections",
    "decimal", "fractions", "string", "functools", "operator", "heapq", "bisect", "textwrap",
}
_BLOCKED_CALLS = {
    "exec", "eval", "compile", "open", "__import__", "input", "breakpoint", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "exit", "quit",
}
_BLOCKED_ATTRS = {
    "system", "popen", "check_output", "check_call", "call", "run", "Popen", "spawn", "remove",
    "rmdir", "unlink", "rmtree", "exec", "eval",
}


def detect_dangerous_python(code: str) -> bool:
    """True if the script needs human approval (non-whitelisted imports, dynamic exec, file/process access)."""
    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in SAFE_IMPORTS for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in SAFE_IMPORTS:
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                return True
            if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTRS:
                return True
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__import__"}:
            return True
    return False


def run_python(code: str, timeout: Optional[int] = None) -> dict:
    """Run code in a separate interpreter with a timeout and a scrubbed environment
    (so API keys in the parent environment are not exposed). This is process isolation,
    not a hardened sandbox: use a container for untrusted production workloads."""
    timeout = timeout or PYTHON_TIMEOUT_SECONDS
    env = {"PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
    if os.name == "nt" and os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1, "timed_out": True}

    return {
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-4000:],
        "returncode": proc.returncode,
        "timed_out": False,
    }
