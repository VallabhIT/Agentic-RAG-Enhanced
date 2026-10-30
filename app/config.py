import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", str(DATA_DIR / "chroma")))
SQLITE_DB_PATH = DATA_DIR / "demo.db"

# ---- Pipeline tunables (all overridable via environment) -------------------
RAG_MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))            # refinement retries after the 1st retrieval
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.5"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "5"))
MAX_TOOL_CHARS = int(os.getenv("MAX_TOOL_CHARS", "12000"))          # cap on tool output sent to the finalize LLM
PYTHON_TIMEOUT_SECONDS = int(os.getenv("PYTHON_TIMEOUT_SECONDS", "10"))
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "6"))              # prior messages used for context
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "40"))


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.1):
    """Return a (cached) chat model. Imports are lazy so tests can run without provider SDKs."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {provider}")
