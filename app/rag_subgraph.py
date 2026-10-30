"""RAG subgraph: retrieve -> judge -> (refine -> retrieve)* -> END.

It does NOT write the final answer. It returns the retrieved documents plus an explicit
`relevant` verdict; the parent graph decides whether to continue with the documents or fall
back to web search, and the finalize node writes the user-facing answer.
"""
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import config, tools
from app.prompts import JUDGE_PROMPT, RAG_REFINE_PROMPT
from app.utils import extract_json


class RAGState(TypedDict, total=False):
    query: str            # canonical (rewritten) query; relevance is judged against this
    search_query: str     # query actually sent to the vector store (changes on refinement)
    retrieved_docs: list[str]
    judge_score: float
    judge_reason: str
    relevant: bool
    retry_count: int
    status: str


def judge_docs(query: str, docs: list[str]) -> dict:
    if not docs:
        return {"relevant": False, "score": 0.0, "reason": "No documents were retrieved."}

    numbered = "\n---\n".join(f"[Doc {i + 1}] {d}" for i, d in enumerate(docs))
    try:
        result = config.get_llm().invoke(JUDGE_PROMPT.format(query=query, docs=numbered))
    except Exception as exc:
        return {"relevant": False, "score": 0.0, "reason": f"Judge call failed: {exc}"}

    parsed = extract_json(str(result.content))
    if not parsed:
        return {"relevant": False, "score": 0.0, "reason": "Could not parse judge output."}
    try:
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    relevant = bool(parsed.get("relevant", False)) and score >= config.RAG_RELEVANCE_THRESHOLD
    return {"relevant": relevant, "score": score, "reason": str(parsed.get("reason", ""))}


def retrieve_node(state: RAGState):
    query = state.get("search_query") or state["query"]
    docs = tools.retrieve_docs(query, k=config.RAG_TOP_K)
    return {"retrieved_docs": docs, "search_query": query, "status": "retrieved"}


def judge_node(state: RAGState):
    result = judge_docs(state["query"], state.get("retrieved_docs", []))
    return {
        "judge_score": result["score"],
        "relevant": result["relevant"],
        "judge_reason": result["reason"],
        "status": "judged",
    }


def refine_node(state: RAGState):
    """Produce a *different* search query and count the retry (this is what bounds the loop)."""
    previous = state.get("search_query") or state["query"]
    new_query = previous
    try:
        out = config.get_llm().invoke(
            RAG_REFINE_PROMPT.format(
                query=state["query"], previous_query=previous,
                reason=state.get("judge_reason") or "no relevant documents found",
            )
        )
        candidate = str(out.content).strip().strip('"').strip()
        if candidate:
            new_query = candidate
    except Exception:
        pass
    return {
        "search_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "status": "refined",
    }


def route_after_judge(state: RAGState):
    if state.get("relevant"):
        return "done"
    if state.get("retry_count", 0) < config.RAG_MAX_RETRIES:
        return "refine"
    return "done"  # retries exhausted and still not relevant -> parent graph falls back to web


@lru_cache(maxsize=1)
def build_rag_subgraph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("refine", refine_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "judge")
    workflow.add_conditional_edges("judge", route_after_judge, {"refine": "refine", "done": END})
    workflow.add_edge("refine", "retrieve")
    return workflow.compile()
