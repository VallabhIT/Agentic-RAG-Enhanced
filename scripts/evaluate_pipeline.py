"""Black-box application evaluation plus agent tool-selection scoring.

This suite measures route accuracy deterministically and uses DeepEval for:
- Answer Relevancy: whether the final response addresses the request.
- Tool Correctness: whether the route/tool choice matches the golden expectation.

Trajectory metrics such as Task Completion and Step Efficiency are intentionally left out until
runtime tracing is wired into the application; adding them without a real trace would be misleading.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.graph import build_graph
from app.state import new_state
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

ROOT = Path(__file__).resolve().parents[1]


def infer_tool_calls(result: dict) -> list[ToolCall]:
    route = result.get("route")
    names = []
    if route == "rag":
        names.append("rag")
        if result.get("web_results") and not result.get("rag_relevant"):
            names.append("web_search")
    elif route == "sql" and result.get("sql_result") is not None:
        names.append("sql")
    elif route == "python" and result.get("python_result") is not None:
        names.append("python")
    elif route == "web":
        names.append("web_search")
    elif route == "direct":
        names.append("direct")
    return [ToolCall(name=name) for name in names]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Evaluate one case id")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evals/pipeline_cases.json").read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
    graph = build_graph()
    model = os.getenv("DEEPEVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    answer_metric = AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True)
    tool_metric = ToolCorrectnessMetric(threshold=0.7, model=model, include_reason=True)
    available = [ToolCall(name=name) for name in ("rag", "sql", "python", "web_search", "direct")]
    route_hits = 0

    for case in cases:
        result = graph.invoke(new_state(case["input"], []), {"recursion_limit": 40})
        actual_route = result.get("route")
        route_hits += int(actual_route == case["expected_route"])
        actual_tools = infer_tool_calls(result)
        expected_tools = [ToolCall(name=case["expected_route"] if case["expected_route"] != "web" else "web_search")]
        test_case = LLMTestCase(
            name=case["id"],
            input=case["input"],
            actual_output=result.get("final_answer", ""),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )
        try:
            answer_metric.measure(test_case)
            answer_score, answer_reason = answer_metric.score, answer_metric.reason
        except Exception as exc:
            answer_score, answer_reason = None, str(exc)
        try:
            tool_metric.measure(test_case)
            tool_score, tool_reason = tool_metric.score, tool_metric.reason
        except Exception as exc:
            tool_score, tool_reason = None, str(exc)
        print(f"\n[{case['id']}] route={actual_route} expected={case['expected_route']}")
        print(f"  final: {result.get('final_answer','')}")
        print(f"  answer_relevancy: {answer_score} — {answer_reason}")
        print(f"  tool_correctness: {tool_score} — {tool_reason}")

    print(f"\nRoute accuracy: {route_hits / max(len(cases), 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
