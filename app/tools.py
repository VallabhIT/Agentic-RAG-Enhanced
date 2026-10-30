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
    return [
        Document(page_content=item["content"], metadata={"title": item["title"], "id": item["id"], "source": "demo"})
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

    # Do not assume an empty collection: Notion/other source syncs may already have populated it.
    docs = load_demo_documents()
    demo_ids = [str(d.metadata["id"]) for d in docs]
    existing_ids = set(vectorstore.get(ids=demo_ids).get("ids") or [])
    missing = [doc for doc in docs if str(doc.metadata["id"]) not in existing_ids]
    if missing:
        vectorstore.add_documents(missing, ids=[str(d.metadata["id"]) for d in missing])
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
    if SQLITE_DB_PATH.exists():
        return SQLITE_DB_PATH
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, spend REAL)")
        cur.executemany(
            "INSERT INTO customers (name, city, spend) VALUES (?, ?, ?)",
            [("Alice", "New York", 2500), ("Bob", "Boston", 1800), ("Charlie", "Chicago", 3200)],
        )
        conn.commit()
    finally:
        conn.close()
    return SQLITE_DB_PATH


def get_schema() -> str:
    """Return CREATE statements for every table, for use in the SQL-generation prompt."""
    db_path = ensure_demo_db()
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return "\n".join(r[0] for r in rows) or "(no tables)"


def run_sql(query: str) -> dict:
    """Execute ONE statement. Read-only statements use a read-only connection (enforced by SQLite);
    anything else must have been approved upstream and runs on a writable connection."""
    db_path = ensure_demo_db()
    read_only = not is_dangerous_sql(query)
    uri = db_path.as_uri() + ("?mode=ro" if read_only else "?mode=rw")
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(query)
        if cur.description:
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchmany(SQL_MAX_ROWS + 1)]
            truncated = len(rows) > SQL_MAX_ROWS
            return {"columns": columns, "rows": rows[:SQL_MAX_ROWS], "row_count": min(len(rows), SQL_MAX_ROWS),
                    "truncated": truncated}
        conn.commit()
        return {"columns": [], "rows": [], "row_count": 0, "rows_affected": cur.rowcount, "truncated": False}
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
