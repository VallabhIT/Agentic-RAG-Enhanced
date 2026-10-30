"""Offline tests: the LLM and all external tools are faked, so no API keys / network needed."""
import pytest

from app import config, graph as graph_mod, rag_subgraph, tools
from app.graph import build_graph
from app.main import run_turn
from app.state import new_state


class Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Dispatches on distinctive phrases in each prompt and records every call."""

    def __init__(self, route="rag", judge=None, rewrite=None, sql="SELECT deliverable FROM client_deliverables WHERE progress_pct > 50",
                 code="print(2 + 2)"):
        self.route, self.judge = route, list(judge or [])
        self.rewrite, self.sql, self.code = rewrite, sql, code
        self.calls = []  # (kind, prompt)

    def invoke(self, prompt):
        p = str(prompt)
        if "You normalize the latest user request" in p:
            kind, out = "rewrite", self.rewrite or p.split("Latest user message:")[1].strip()
        elif "route classifier" in p:
            kind, out = "router", '{"route": "%s"}' % self.route
        elif "evidence-sufficiency judge" in p:
            kind = "judge"
            rel = self.judge.pop(0) if self.judge else False
            out = '{"relevant": %s, "score": %s, "reason": "r"}' % (str(rel).lower(), 0.9 if rel else 0.1)
        elif "better retrieval query" in p:
            kind, out = "refine", "refined query %d" % sum(1 for k, _ in self.calls if k == "refine")
        elif "expert SQLite analyst" in p:
            kind, out = "sql_gen", "```sql\n%s\n```" % self.sql
        elif "expert Python programmer" in p:
            kind, out = "py_gen", "```python\n%s\n```" % self.code
        elif "friendly and concise assistant" in p:
            kind, out = "direct", "Hello! How can I help?"
        elif "final reasoning step" in p:
            kind, out = "finalize", "FINAL ANSWER"
        else:
            raise AssertionError("unexpected prompt: " + p[:80])
        self.calls.append((kind, p))
        return Resp(out)

    def kinds(self):
        return [k for k, _ in self.calls]

    def prompt(self, kind):
        return next(p for k, p in self.calls if k == kind)


@pytest.fixture
def setup(monkeypatch):
    def _make(**kw):
        llm = FakeLLM(**kw)
        monkeypatch.setattr(config, "get_llm", lambda temperature=0.1: llm)
        rec = {"retrieve": [], "web": [], "sql": [], "python": []}
        monkeypatch.setattr(tools, "retrieve_docs", lambda q, k=5: rec["retrieve"].append(q) or ["policy doc text"])
        monkeypatch.setattr(tools, "web_search", lambda q, max_results=5: rec["web"].append(q) or
                            [{"title": "T", "href": "https://x.test", "body": "snippet", "id": "abc123"}])
        monkeypatch.setattr(tools, "get_schema", lambda: "CREATE TABLE client_deliverables (client, project, deliverable, status, priority, progress_pct)")
        monkeypatch.setattr(tools, "run_sql", lambda q: rec["sql"].append(q) or {"columns": ["deliverable"], "rows": [["API Integration Endpoints"]]})
        monkeypatch.setattr(tools, "run_python", lambda c: rec["python"].append(c) or
                            {"stdout": "4\n", "stderr": "", "returncode": 0, "timed_out": False})
        return llm, rec, build_graph()
    return _make


def invoke(g, text):
    return g.invoke(new_state(text), {"recursion_limit": 40})


# ---- 2. conversational queries ------------------------------------------------
def test_smalltalk_skips_pipeline(setup):
    llm, rec, g = setup()
    out = invoke(g, "hi")
    assert out["final_answer"] == "Hello! How can I help?"
    assert llm.kinds() == ["direct"]                      # no rewrite, router, judge, finalize-with-tools
    assert not any(rec.values())                          # no tool touched


def test_thanks_is_conversational(setup):
    llm, rec, g = setup()
    assert invoke(g, "Thanks!")["route"] == "direct"


# ---- 5. rewrite first, original preserved ---------------------------------------
def test_rewrite_happens_first_and_original_is_preserved(setup):
    llm, rec, g = setup(route="rag", judge=[True], rewrite="What is the company policy?")
    out = invoke(g, "wats teh compny polcy")
    assert llm.kinds()[:2] == ["rewrite", "router"]
    assert "What is the company policy?" in llm.prompt("router")
    assert rec["retrieve"] == ["What is the company policy?"]
    assert out["original_query"] == "wats teh compny polcy"
    assert out["rewritten_query"] == "What is the company policy?"
    assert "wats teh compny polcy" in llm.prompt("finalize")   # original available to the final answer


# ---- 6. RAG -> web fallback on real relevance ------------------------------------
def test_relevant_rag_skips_web(setup):
    llm, rec, g = setup(route="rag", judge=[True])
    out = invoke(g, "what does the product do")
    assert rec["web"] == [] and len(rec["retrieve"]) == 1
    assert out["rag_relevant"] and "policy doc text" in llm.prompt("finalize")


def test_irrelevant_after_retries_falls_back_to_web(setup):
    llm, rec, g = setup(route="rag", judge=[False, False, False])
    out = invoke(g, "who won the match yesterday")
    assert len(rec["retrieve"]) == 1 + config.RAG_MAX_RETRIES            # bounded: no infinite loop
    assert rec["retrieve"][1] == "refined query 0"                       # retries use the refined query
    assert out["retry_count"] == config.RAG_MAX_RETRIES
    assert len(rec["web"]) == 1 and rec["web"][0] == out["rewritten_query"]
    fp = llm.prompt("finalize")
    assert "snippet" in fp and "policy doc text" not in fp               # irrelevant docs are not passed on
    assert "fallback" in fp


def test_relevant_on_retry_does_not_use_web(setup):
    llm, rec, g = setup(route="rag", judge=[False, True])
    out = invoke(g, "q")
    assert len(rec["retrieve"]) == 2 and rec["web"] == [] and out["rag_relevant"]


# ---- 1. finalize consolidates raw tool output -----------------------------------
def test_finalize_receives_full_raw_web_results(setup):
    llm, rec, g = setup(route="web")
    out = invoke(g, "latest news on openai")
    fp = llm.prompt("finalize")
    assert "abc123" in fp and "https://x.test" in fp       # full structure given to the LLM
    assert "Ground claims in the actual tool output" in fp                          # ...with instruction to consolidate
    assert out["final_answer"] == "FINAL ANSWER"           # user sees LLM output, not raw dict


# ---- 4. generated SQL / Python executed instead of the NL query ----------------
def test_sql_is_generated_stored_and_executed(setup):
    llm, rec, g = setup(route="sql", sql="SELECT deliverable FROM client_deliverables WHERE progress_pct > 50")
    out = invoke(g, "which deliverables have progress above 50 percent?")
    assert out["sql_query"] == "SELECT deliverable FROM client_deliverables WHERE progress_pct > 50"
    assert rec["sql"] == [out["sql_query"]]                # NOT the natural-language question
    assert out["sql_result"]["rows"] == [["API Integration Endpoints"]]
    assert out["approval_status"] == "not_required"


def test_python_is_generated_stored_and_executed(setup):
    llm, rec, g = setup(route="python", code="print(2 + 2)")
    out = invoke(g, "compute 2 + 2")
    assert out["python_code"] == "print(2 + 2)" and rec["python"] == ["print(2 + 2)"]
    assert out["python_result"]["stdout"] == "4\n"


# ---- 3. approval state: no restart, no re-check ----------------------------------
def test_sql_approval_resumes_without_restart(setup, monkeypatch):
    llm, rec, g = setup(route="sql", sql="DELETE FROM client_deliverables WHERE id = 1")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    out = run_turn(g, "delete a deliverable record", [])
    assert rec["sql"] == ["DELETE FROM client_deliverables WHERE id = 1"]
    assert llm.kinds().count("rewrite") == 1               # graph did NOT start over
    assert llm.kinds().count("router") == 1
    assert llm.kinds().count("sql_gen") == 1               # SQL not regenerated
    assert out["approval_status"] == "granted" and out["pending_action"] == ""
    assert out["final_answer"] == "FINAL ANSWER"


def test_python_approval_resumes_without_restart(setup, monkeypatch):
    llm, rec, g = setup(route="python", code="import os\nprint(os.getcwd())")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    out = run_turn(g, "show cwd", [])
    assert rec["python"] == ["import os\nprint(os.getcwd())"]
    assert llm.kinds().count("py_gen") == 1 and llm.kinds().count("router") == 1


def test_denied_approval_executes_nothing(setup, monkeypatch):
    llm, rec, g = setup(route="sql", sql="DROP TABLE client_deliverables")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    out = run_turn(g, "drop it", [])
    assert rec["sql"] == [] and "denied" in out["final_answer"]


def test_execute_node_refuses_tampered_payload(setup):
    llm, rec, g = setup()
    st = new_state("x")
    st.update(route="sql", sql_query="DROP TABLE client_deliverables", pending_action="sql",
              pending_payload="SELECT 1", approval_status="granted")
    out = g.invoke(st)
    assert rec["sql"] == [] and "Blocked" in (out["error"] or "")


# ---- other fixes -------------------------------------------------------------------
def test_garbage_router_output_defaults_to_rag_not_direct(setup):
    llm, rec, g = setup(route="banana", judge=[True])
    out = invoke(g, "tell me about the platform")
    assert out["route"] == "rag"


def test_prompts_format_without_keyerror():
    from app import prompts
    prompts.ROUTER_PROMPT.format(query="q", sql_schema="CREATE TABLE client_deliverables (...)")
    prompts.JUDGE_PROMPT.format(query="q", docs="d")


def test_sql_safety_classifier():
    assert not tools.is_dangerous_sql("SELECT * FROM customers")
    assert not tools.is_dangerous_sql("-- c\nWITH a AS (SELECT 1) SELECT * FROM a")
    for bad in ["DROP TABLE t", "delete from t", "SELECT 1; DROP TABLE t", "PRAGMA writable_schema=1",
                "ATTACH DATABASE 'x' AS y", "", "WITH a AS (SELECT 1) DELETE FROM t"]:
        assert tools.is_dangerous_sql(bad), bad


def test_python_safety_classifier():
    assert not tools.detect_dangerous_python("import math\nprint(math.sqrt(4))")
    for bad in ["import os", "import subprocess", "open('f').read()", "eval('1')",
                "print(().__class__.__bases__)", "from os import system", "getattr(x, 'y')"]:
        assert tools.detect_dangerous_python(bad), bad


def test_real_sql_and_python_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SQLITE_DB_PATH", tmp_path / "t.db")
    res = tools.run_sql("SELECT deliverable FROM client_deliverables WHERE client = 'Acme Retail' ORDER BY deliverable")
    assert len(res["rows"]) == 3
    assert all(row[0] for row in res["rows"])
    with pytest.raises(Exception):                         # read-only connection blocks writes
        tools.run_sql("SELECT 1; DELETE FROM client_deliverables")
    monkeypatch.setenv("SECRET_KEY", "hunter2")
    r = tools.run_python("import os\nprint(os.environ.get('SECRET_KEY'))")
    assert r["stdout"].strip() == "None"                   # parent env is not leaked
    assert tools.run_python("while True: pass", timeout=1)["timed_out"]
