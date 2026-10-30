from typing import Annotated, Any, Literal, Optional, TypedDict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

ApprovalStatus = Literal["not_required", "pending", "granted", "denied"]


class AgentState(TypedDict, total=False):
    # ---- conversation ----
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str          # exactly what the user typed; never overwritten
    rewritten_query: str         # canonical query used by every downstream node

    # ---- routing ----
    route: str                   # rag | sql | python | web | direct

    # ---- RAG ----
    retrieved_docs: list[str]
    rag_relevant: bool
    judge_score: float
    judge_reason: str
    retry_count: int             # RAG refinement retries actually used
    search_query: str            # last query sent to the vector store

    # ---- web ----
    web_results: list[dict]      # raw structured results (title/href/body/...)

    # ---- generated code + execution output (kept for audit/debug) ----
    sql_query: str
    sql_result: Optional[dict]
    python_code: str
    python_result: Optional[dict]

    # ---- human approval ----
    approval_status: ApprovalStatus
    approval_required: bool
    approval_message: str
    pending_action: str          # "sql" | "python" | ""   (what is waiting for approval)
    pending_payload: str         # exact code that was shown to the human

    # ---- output ----
    final_answer: str
    status: str
    error: Optional[str]


def new_state(user_input: str, history: Optional[list[BaseMessage]] = None) -> AgentState:
    """Build a fresh state for a new user turn."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [*(history or []), HumanMessage(content=user_input)],
        "original_query": user_input,
        "rewritten_query": "",
        "route": "",
        "retrieved_docs": [],
        "rag_relevant": False,
        "judge_score": 0.0,
        "judge_reason": "",
        "retry_count": 0,
        "search_query": "",
        "web_results": [],
        "sql_query": "",
        "sql_result": None,
        "python_code": "",
        "python_result": None,
        "approval_status": "not_required",
        "approval_required": False,
        "approval_message": "",
        "pending_action": "",
        "pending_payload": "",
        "final_answer": "",
        "status": "new",
        "error": None,
    }
