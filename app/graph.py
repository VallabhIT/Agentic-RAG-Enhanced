"""Main agent graph.

 entry_gate ─┬─ (normal turn) ─────────► rewrite ─► router ─┬─ rag ──► [relevant?] ─ yes ─► finalize
             │                              │               │             └─ no ─► web_search ─► finalize
             │                              └─ smalltalk ─► finalize (direct)
             │                                              ├─ web ──► web_search ─► finalize
             │                                              ├─ sql ──► sql_generate ─► sql_guard ─┬─ safe ─► sql_execute ─► finalize
             │                                              │                                     └─ needs approval ─► END (pause)
             │                                              └─ python ─► python_generate ─► python_guard ─ (same pattern)
             ├─ approval_status == granted ─► sql_execute / python_execute   (skips rewrite, routing and the danger check)
             └─ approval_status == denied  ─► finalize
"""
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app import config, tools
from app.prompts import (
    DIRECT_PROMPT,
    FINALIZE_PROMPT,
    PYTHON_GENERATE_PROMPT,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    SQL_GENERATE_PROMPT,
)
from app.rag_subgraph import build_rag_subgraph
from app.state import AgentState
from app.utils import extract_json, format_history, is_smalltalk, strip_code_fences, to_json

VALID_ROUTES = {"rag", "sql", "python", "web", "direct"}


# --------------------------------------------------------------------------- #
# Entry gate: decides whether this invocation is a fresh turn or an approval resume
# --------------------------------------------------------------------------- #
def entry_gate(state: AgentState) -> str:
    pending = state.get("pending_action")
    status = state.get("approval_status", "not_required")
    if pending in {"sql", "python"}:
        if status == "granted":
            return "sql_execute" if pending == "sql" else "python_execute"
        if status == "denied":
            return "finalize"
    return "rewrite"


# --------------------------------------------------------------------------- #
# 1. Query rewriting (first real step) + routing
# --------------------------------------------------------------------------- #
def rewrite_node(state: AgentState):
    original = state["original_query"]

    if is_smalltalk(original):  # nothing to normalise; skip an LLM call
        return {"rewritten_query": original, "route": "direct", "status": "rewritten"}

    rewritten = original
    try:
        history = format_history(state.get("messages", []), config.HISTORY_WINDOW)
        out = config.get_llm(0.0).invoke(REWRITE_PROMPT.format(history=history, query=original))
        candidate = str(out.content).strip().strip('"').strip()
        # Guard against empty or runaway rewrites.
        if candidate and len(candidate) <= max(400, 4 * len(original)):
            rewritten = candidate
    except Exception:
        pass  # fall back to the original query
    return {"rewritten_query": rewritten, "route": "", "status": "rewritten"}


def route_after_rewrite(state: AgentState) -> str:
    return "finalize" if state.get("route") == "direct" else "router"


def router_node(state: AgentState):
    query = state["rewritten_query"]
    route = None
    try:
        out = config.get_llm(0.0).invoke(ROUTER_PROMPT.format(query=query))
        parsed = extract_json(str(out.content))
        candidate = str((parsed or {}).get("route", "")).strip().lower()
        if candidate in VALID_ROUTES:
            route = candidate
    except Exception:
        route = None
    # Unknown/unparseable output -> search knowledge (which falls back to web) instead of guessing.
    return {"route": route or "rag", "status": "routed"}


def route_after_router(state: AgentState) -> str:
    return {
        "rag": "rag", "sql": "sql_generate", "python": "python_generate",
        "web": "web_search", "direct": "finalize",
    }.get(state.get("route", "rag"), "rag")


# --------------------------------------------------------------------------- #
# 2. RAG, with web fallback driven by actual relevance
# --------------------------------------------------------------------------- #
def rag_node(state: AgentState):
    result = build_rag_subgraph().invoke(
        {"query": state["rewritten_query"], "search_query": state["rewritten_query"], "retry_count": 0}
    )
    relevant = bool(result.get("relevant", False))
    return {
        "retrieved_docs": result.get("retrieved_docs", []),
        "rag_relevant": relevant,
        "judge_score": result.get("judge_score", 0.0),
        "judge_reason": result.get("judge_reason", ""),
        "retry_count": result.get("retry_count", 0),
        "search_query": result.get("search_query", state["rewritten_query"]),
        "status": "rag_relevant" if relevant else "rag_insufficient",
    }


def route_after_rag(state: AgentState) -> str:
    return "finalize" if state.get("rag_relevant") else "web_search"


def web_search_node(state: AgentState):
    results = tools.web_search(state["rewritten_query"], max_results=config.WEB_MAX_RESULTS)
    return {"web_results": results, "status": "web_complete"}


# --------------------------------------------------------------------------- #
# 3. SQL: generate -> guard -> execute
# --------------------------------------------------------------------------- #
def sql_generate_node(state: AgentState):
    try:
        schema = tools.get_schema()
        out = config.get_llm(0.0).invoke(
            SQL_GENERATE_PROMPT.format(schema=schema, query=state["rewritten_query"])
        )
        sql = strip_code_fences(str(out.content)).strip()
    except Exception as exc:
        return {"sql_query": "", "error": f"SQL generation failed: {exc}", "status": "sql_error"}
    if not sql:
        return {"sql_query": "", "error": "The model did not produce a SQL query.", "status": "sql_error"}
    return {"sql_query": sql, "error": None, "status": "sql_generated"}


def sql_guard_node(state: AgentState):
    sql = state.get("sql_query", "")
    if not sql:
        return {"status": "sql_error"}
    if tools.is_dangerous_sql(sql):
        return {
            "approval_required": True,
            "approval_status": "pending",
            "pending_action": "sql",
            "pending_payload": sql,
            "approval_message": f"Approve this SQL before execution:\n\n{sql}",
            "status": "awaiting_approval",
        }
    return {"approval_required": False, "approval_status": "not_required", "status": "sql_validated"}


def route_after_guard(state: AgentState) -> str:
    if state.get("status") == "awaiting_approval":
        return "pause"
    if state.get("status") == "sql_error" or state.get("status") == "python_error":
        return "finalize"
    return "execute"


def sql_execute_node(state: AgentState):
    sql = state["sql_query"]
    needs_approval = tools.is_dangerous_sql(sql)
    approved = state.get("approval_status") == "granted" and state.get("pending_payload") == sql
    if needs_approval and not approved:  # defense in depth: never run unapproved risky SQL
        return {"error": "Blocked: this SQL requires human approval that was not granted for this exact statement.",
                "status": "sql_error"}
    try:
        result = tools.run_sql(sql)
        return {"sql_result": result, "error": None, "status": "sql_complete",
                "approval_required": False, "pending_action": "", "pending_payload": ""}
    except Exception as exc:
        return {"sql_result": None, "error": f"SQL execution failed: {exc}", "status": "sql_error",
                "approval_required": False, "pending_action": "", "pending_payload": ""}


# --------------------------------------------------------------------------- #
# 4. Python: generate -> guard -> execute
# --------------------------------------------------------------------------- #
def python_generate_node(state: AgentState):
    try:
        out = config.get_llm(0.0).invoke(PYTHON_GENERATE_PROMPT.format(query=state["rewritten_query"]))
        code = strip_code_fences(str(out.content)).strip()
    except Exception as exc:
        return {"python_code": "", "error": f"Python generation failed: {exc}", "status": "python_error"}
    if not code:
        return {"python_code": "", "error": "The model did not produce any code.", "status": "python_error"}
    return {"python_code": code, "error": None, "status": "python_generated"}


def python_guard_node(state: AgentState):
    code = state.get("python_code", "")
    if not code:
        return {"status": "python_error"}
    if tools.detect_dangerous_python(code):
        return {
            "approval_required": True,
            "approval_status": "pending",
            "pending_action": "python",
            "pending_payload": code,
            "approval_message": f"Approve this Python code before execution:\n\n{code}",
            "status": "awaiting_approval",
        }
    return {"approval_required": False, "approval_status": "not_required", "status": "python_validated"}


def python_execute_node(state: AgentState):
    code = state["python_code"]
    needs_approval = tools.detect_dangerous_python(code)
    approved = state.get("approval_status") == "granted" and state.get("pending_payload") == code
    if needs_approval and not approved:
        return {"error": "Blocked: this code requires human approval that was not granted for this exact script.",
                "status": "python_error"}
    try:
        result = tools.run_python(code)
        err = result["stderr"] if result["returncode"] != 0 else None
        return {"python_result": result, "error": err, "status": "python_complete",
                "approval_required": False, "pending_action": "", "pending_payload": ""}
    except Exception as exc:
        return {"python_result": None, "error": f"Python execution failed: {exc}", "status": "python_error",
                "approval_required": False, "pending_action": "", "pending_payload": ""}


# --------------------------------------------------------------------------- #
# 5. Finalize: conversational reply OR consolidation/reasoning over tool results
# --------------------------------------------------------------------------- #
def _build_tool_context(state: AgentState) -> tuple[str, str]:
    """Return (source_note, tool_context) describing exactly what the tools produced."""
    limit = config.MAX_TOOL_CHARS
    route = state.get("route", "")
    parts: list[str] = []
    notes: list[str] = []

    if state.get("rag_relevant") and state.get("retrieved_docs"):
        docs = "\n---\n".join(f"[Doc {i + 1}] {d}" for i, d in enumerate(state["retrieved_docs"]))
        parts.append(f"## Internal knowledge-base documents\n{docs}")
        notes.append("Answer from the internal knowledge base.")
    elif route == "rag":
        notes.append(
            "The internal knowledge base did not contain relevant information"
            f" (judge: {state.get('judge_reason') or 'n/a'}), so web search was used as a fallback."
            " Say briefly that internal documents did not cover this."
        )

    if state.get("web_results") is not None and route in {"rag", "web"} and not state.get("rag_relevant"):
        parts.append("## Web search results (raw)\n" + to_json(state.get("web_results", []), limit))
        if route == "web":
            notes.append("Answer from live web search.")

    if state.get("sql_query"):
        block = f"## SQL query that was executed\n{state['sql_query']}"
        if state.get("sql_result") is not None:
            block += "\n\n## SQL result (raw)\n" + to_json(state["sql_result"], limit)
        parts.append(block)
        notes.append("Answer from a generated SQL query run against the database.")

    if state.get("python_code"):
        block = f"## Python script that was executed\n{state['python_code']}"
        if state.get("python_result") is not None:
            block += "\n\n## Python result (raw)\n" + to_json(state["python_result"], limit)
        parts.append(block)
        notes.append("Answer from a generated Python script that was executed.")

    if state.get("error"):
        parts.append(f"## Tool error\n{state['error']}")

    context = "\n\n".join(parts) if parts else "(no tool output was produced)"
    return " ".join(notes) or "No tools were used.", context[: limit * 2]


def finalize_node(state: AgentState):
    original = state.get("original_query", "")
    route = state.get("route", "")
    ai = lambda text: {"final_answer": text, "status": "finalized", "messages": [AIMessage(content=text)]}

    # Human said no: do not call the LLM, do not execute anything.
    if state.get("approval_status") == "denied":
        return ai("Execution was denied by the human reviewer, so nothing was run.")

    llm = config.get_llm()
    print(f'Route: {route}')

    # Conversational turn: there are no tool results to expect.
    if route == "direct":
        try:
            history = format_history(state.get("messages", []), config.HISTORY_WINDOW)
            out = llm.invoke(DIRECT_PROMPT.format(history=history, query=original))
            return ai(str(out.content).strip() or "Hello! How can I help you today?")
        except Exception:
            return ai("Hello! How can I help you today?")

    source_note, tool_context = _build_tool_context(state)
    try:
        out = llm.invoke(
            FINALIZE_PROMPT.format(
                original_query=original,
                rewritten_query=state.get("rewritten_query", original),
                source_note=source_note,
                tool_context=tool_context,
            )
        )
        answer = str(out.content).strip()
        if not answer:
            raise ValueError("empty answer")
        return ai(answer)
    except Exception as exc:
        return ai(f"I ran the tools but could not produce a final answer ({exc}). Please try again.")


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    wf = StateGraph(AgentState)

    wf.add_node("rewrite", rewrite_node)
    wf.add_node("router", router_node)
    wf.add_node("rag", rag_node)
    wf.add_node("web_search", web_search_node)
    wf.add_node("sql_generate", sql_generate_node)
    wf.add_node("sql_guard", sql_guard_node)
    wf.add_node("sql_execute", sql_execute_node)
    wf.add_node("python_generate", python_generate_node)
    wf.add_node("python_guard", python_guard_node)
    wf.add_node("python_execute", python_execute_node)
    wf.add_node("finalize", finalize_node)

    wf.set_conditional_entry_point(
        entry_gate,
        {"rewrite": "rewrite", "sql_execute": "sql_execute", "python_execute": "python_execute",
         "finalize": "finalize"},
    )
    wf.add_conditional_edges("rewrite", route_after_rewrite, {"router": "router", "finalize": "finalize"})
    wf.add_conditional_edges(
        "router", route_after_router,
        {"rag": "rag", "sql_generate": "sql_generate", "python_generate": "python_generate",
         "web_search": "web_search", "finalize": "finalize"},
    )

    wf.add_conditional_edges("rag", route_after_rag, {"finalize": "finalize", "web_search": "web_search"})
    wf.add_edge("web_search", "finalize")

    wf.add_edge("sql_generate", "sql_guard")
    wf.add_conditional_edges(
        "sql_guard", route_after_guard,
        {"execute": "sql_execute", "finalize": "finalize", "pause": END},
    )
    wf.add_edge("sql_execute", "finalize")

    wf.add_edge("python_generate", "python_guard")
    wf.add_conditional_edges(
        "python_guard", route_after_guard,
        {"execute": "python_execute", "finalize": "finalize", "pause": END},
    )
    wf.add_edge("python_execute", "finalize")

    wf.add_edge("finalize", END)
    return wf.compile()
