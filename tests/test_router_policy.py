from __future__ import annotations

import json
from pathlib import Path

from app import tools
from app.graph import router_node
from app.prompts import ROUTER_PROMPT


ROOT = Path(__file__).resolve().parents[1]


def test_golden_route_hints():
    cases = json.loads((ROOT / "evals" / "routing_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        hint = tools.route_hint(case["query"])
        if case["expected_route"] in {"sql", "web", "python"}:
            assert hint == case["expected_route"], case["query"]
        else:
            # RAG and direct intents are deliberately left to the graph's existing
            # smalltalk gate or schema-grounded LLM router.
            assert hint is None, case["query"]


def test_obvious_tracker_query_bypasses_router_llm():
    state = {"rewritten_query": "Show all blocked deliverables for Northstar Logistics"}
    result = router_node(state)
    assert result["route"] == "sql"
    assert result["route_reason"] == "deterministic_intent_hint"


def test_current_tracker_query_is_sql_not_web():
    assert tools.route_hint("What is the latest status of the Northstar deliverables?") == "sql"


def test_agent_concept_query_is_not_sql():
    assert tools.route_hint("Explain how an agent should handle a blocked tool call") is None
    assert tools.route_hint("What does a blocked deliverable mean in a project tracker?") is None
    assert tools.route_hint("How do I evaluate RAG faithfulness and citation coverage?") is None


def test_router_prompt_has_live_schema_and_source_boundary():
    rendered = ROUTER_PROMPT.format(
        query="Who owns the blocked deliverable?",
        sql_schema="CREATE TABLE client_deliverables (client TEXT, owner TEXT, status TEXT)",
    )
    assert "client_deliverables" in rendered
    assert "explanations or internal engineering knowledge" in rendered
    assert "latest" in rendered
