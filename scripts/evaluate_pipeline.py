
"""Pipeline evaluation using actual top-level LangGraph execution paths.

Measures:
- Exact execution-path match
- Execution-node precision and recall
- Execution-node ordering
- Answer relevancy (DeepEval, optional)

Routing accuracy is evaluated separately by evaluate_routing.py.

Important:
The RAG subgraph is invoked synchronously inside the top-level `rag` node.
Therefore, this evaluator captures `rag` as one top-level execution step.
It does not capture the nested retrieve/judge/refine trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from app.graph import build_graph
from app.state import new_state


ROOT = Path(__file__).resolve().parents[1]

# Only include nodes that represent actual agent execution.
# Deliberately exclude rewrite, router and finalize.
EXECUTION_NODES = {
    "rag",
    "web_search",
    "sql_generate",
    "sql_guard",
    "sql_execute",
    "python_generate",
    "python_guard",
    "python_execute",
}


def capture_execution_path(
    graph: Any,
    query: str,
) -> tuple[dict, list[str]]:
    """Run the graph once and capture its actual top-level node execution path.

    `stream_mode="updates"` yields updates keyed by the node that produced them.
    The state updates are accumulated so the final answer and route are available
    without invoking the graph a second time.
    """

    accumulated_state: dict = {}
    execution_path: list[str] = []

    events = graph.stream(
        new_state(query, []),
        config={"recursion_limit": 40},
        stream_mode="updates",
    )

    for event in events:
        if not isinstance(event, dict):
            continue

        for node_name, node_update in event.items():

            # Capture only meaningful execution nodes.
            if node_name in EXECUTION_NODES:
                execution_path.append(node_name)

            # Accumulate node state updates.
            if isinstance(node_update, dict):
                accumulated_state.update(node_update)

    return accumulated_state, execution_path


def get_expected_path(case: dict) -> list[str]:
    """Read the expected path, supporting the old dataset field name."""

    expected = case.get(
        "expected_execution_path",
        case.get("expected_tool_path", []),
    )

    if not isinstance(expected, list):
        raise ValueError(
            f"Expected execution path must be a list in case {case.get('id')}"
        )

    expected = [str(node) for node in expected]

    # Direct responses do not invoke an execution node.
    # Support the old synthetic ["direct"] expectation for compatibility.
    if case.get("expected_route") == "direct" and expected == ["direct"]:
        return []

    return expected


def lcs_length(a: list[str], b: list[str]) -> int:
    """Calculate the longest common subsequence length."""

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    return dp[-1][-1]


def counter_difference(
    first: list[str],
    second: list[str],
) -> list[str]:
    """Return occurrences in first that are not matched in second."""

    difference = Counter(first) - Counter(second)
    return list(difference.elements())


def calculate_path_metrics(
    expected: list[str],
    actual: list[str],
) -> dict:
    """Calculate execution-path quality metrics."""

    expected_counts = Counter(expected)
    actual_counts = Counter(actual)

    common = sum((expected_counts & actual_counts).values())

    if not expected and not actual:
        precision = 1.0
        recall = 1.0
        order_score = 1.0

    else:
        precision = common / len(actual) if actual else 0.0
        recall = common / len(expected) if expected else 0.0

        denominator = max(len(expected), len(actual))
        order_score = (
            lcs_length(expected, actual) / denominator
            if denominator
            else 1.0
        )

    return {
        "exact_match": expected == actual,
        "precision": precision,
        "recall": recall,
        "order_score": order_score,
        "missing_nodes": counter_difference(expected, actual),
        "unexpected_nodes": counter_difference(actual, expected),
    }


def get_deepeval_model():
    """Create a DeepEval-compatible OpenRouter model."""

    from deepeval.models import OpenAIModel

    return OpenAIModel(
        model=os.getenv(
            "OPENROUTER_EVALUATION_MODEL",
            "google/gemini-2.5-flash",
        ),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


def evaluate_answer_relevancy(
    query: str,
    answer: str,
    model: Any,
) -> tuple[float | None, str]:
    """Evaluate the final answer with DeepEval."""

    if not answer:
        return None, "No final answer was produced."

    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=model,
            include_reason=True,
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
        )

        metric.measure(test_case)

        return metric.score, metric.reason or ""

    except Exception as exc:
        return None, f"Answer relevancy evaluation failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate actual LangGraph execution paths."
    )

    parser.add_argument(
        "--case",
        help="Evaluate one case ID only.",
    )

    parser.add_argument(
        "--skip-answer-metric",
        action="store_true",
        help="Skip the DeepEval Answer Relevancy metric.",
    )

    args = parser.parse_args()

    cases_path = ROOT / "evals" / "pipeline_cases.json"

    cases = json.loads(
        cases_path.read_text(encoding="utf-8")
    )

    if args.case:
        cases = [
            case for case in cases
            if case["id"] == args.case
        ]

        if not cases:
            print(f"No evaluation case found with ID: {args.case}")
            return 1

    graph = build_graph()

    # Path evaluation works independently of DeepEval/API availability.
    deepeval_model = None

    if not args.skip_answer_metric:
        if not os.getenv("OPENROUTER_API_KEY"):
            print(
                "OPENROUTER_API_KEY is not configured. "
                "Skipping Answer Relevancy; path evaluation will continue."
            )
        else:
            try:
                deepeval_model = get_deepeval_model()
            except Exception as exc:
                print(
                    f"Could not initialize DeepEval model: {exc}. "
                    "Path evaluation will continue."
                )

    results = []

    for case in cases:
        case_id = case["id"]
        query = case["input"]

        expected_path = get_expected_path(case)

        # Run the application exactly once.
        result, actual_path = capture_execution_path(
            graph,
            query,
        )

        path_scores = calculate_path_metrics(
            expected_path,
            actual_path,
        )

        answer = str(result.get("final_answer", "") or "")
        actual_route = result.get("route", "unknown")

        answer_score = None
        answer_reason = "Skipped."

        if deepeval_model is not None:
            answer_score, answer_reason = evaluate_answer_relevancy(
                query,
                answer,
                deepeval_model,
            )

        results.append({
            "id": case_id,
            "route": actual_route,
            "expected_path": expected_path,
            "actual_path": actual_path,
            **path_scores,
            "answer_relevancy": answer_score,
        })

        print(f"\n{'=' * 65}")
        print(f"Case: {case_id}")
        print(f"Route: {actual_route}")
        print(f"Expected path: {expected_path}")
        print(f"Actual path:   {actual_path}")

        print(
            f"Exact path match: {path_scores['exact_match']}"
        )

        print(
            f"Path precision:   {path_scores['precision']:.3f}"
        )

        print(
            f"Path recall:      {path_scores['recall']:.3f}"
        )

        print(
            f"Path order score: {path_scores['order_score']:.3f}"
        )

        if path_scores["missing_nodes"]:
            print(
                f"Missing nodes: {path_scores['missing_nodes']}"
            )

        if path_scores["unexpected_nodes"]:
            print(
                f"Unexpected nodes: {path_scores['unexpected_nodes']}"
            )

        print(f"Final answer: {answer}")

        print(
            f"Answer relevancy: {answer_score} — {answer_reason}"
        )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    total = len(results)

    if not total:
        print("No evaluation cases were executed.")
        return 1

    exact_accuracy = sum(
        r["exact_match"] for r in results
    ) / total

    mean_precision = sum(
        r["precision"] for r in results
    ) / total

    mean_recall = sum(
        r["recall"] for r in results
    ) / total

    mean_order_score = sum(
        r["order_score"] for r in results
    ) / total

    answer_scores = [
        r["answer_relevancy"]
        for r in results
        if r["answer_relevancy"] is not None
    ]

    print(f"\n{'=' * 65}")
    print("PIPELINE EVALUATION SUMMARY")
    print(f"{'=' * 65}")

    print(f"Total cases:             {total}")
    print(f"Exact path accuracy:     {exact_accuracy:.3f}")
    print(f"Mean path precision:     {mean_precision:.3f}")
    print(f"Mean path recall:        {mean_recall:.3f}")
    print(f"Mean path order score:   {mean_order_score:.3f}")

    if answer_scores:
        mean_answer_relevancy = (
            sum(answer_scores) / len(answer_scores)
        )
        print(
            f"Mean answer relevancy:   {mean_answer_relevancy:.3f}"
        )
    else:
        print("Mean answer relevancy:   Not measured")

    print(
        f"Exact path matches:      "
        f"{sum(r['exact_match'] for r in results)}/{total}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
