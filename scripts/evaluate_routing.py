"""Live evaluation of the router against the versioned golden query set.

Run after configuring the existing LLM provider:
    python -m scripts.evaluate_routing

The report distinguishes deterministic short-circuits from LLM decisions. It measures
routing only; it does not claim to score answer quality or retrieval faithfulness.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app import config, tools
from app.prompts import ROUTER_PROMPT
from app.utils import extract_json, is_smalltalk

ROOT = Path(__file__).resolve().parents[1]
VALID_ROUTES = {"rag", "sql", "python", "web", "direct"}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def evaluate() -> dict:
    cases = json.loads((ROOT / "evals" / "routing_cases.json").read_text(encoding="utf-8"))
    schema = tools.get_schema()
    llm = config.get_llm(0.0)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latency_ms: list[float] = []
    deterministic_count = 0
    llm_count = 0
    invalid_count = 0
    correct_count = 0
    results = []

    for case in cases:
        query = case["query"]
        expected = case["expected_route"]
        start = time.perf_counter()

        if is_smalltalk(query):
            predicted = "direct"
            method = "smalltalk_guard"
            deterministic_count += 1
        else:
            hint = tools.route_hint(query)
            if hint in VALID_ROUTES:
                predicted = hint
                method = "deterministic_hint"
                deterministic_count += 1
            else:
                method = "llm_router"
                llm_count += 1
                try:
                    response = llm.invoke(
                        ROUTER_PROMPT.format(query=query, sql_schema=schema)
                    )
                    parsed = extract_json(str(response.content))
                    predicted = str((parsed or {}).get("route", "")).strip().lower()
                    if predicted not in VALID_ROUTES:
                        predicted = "invalid"
                        invalid_count += 1
                except Exception:
                    predicted = "invalid"
                    invalid_count += 1

        elapsed = (time.perf_counter() - start) * 1000
        latency_ms.append(elapsed)
        confusion[expected][predicted] += 1
        correct = predicted == expected
        correct_count += int(correct)
        results.append({
            "expected": expected,
            "predicted": predicted,
            "method": method,
            "correct": correct,
            "latency_ms": round(elapsed, 2),
        })

    total = len(cases)
    return {
        "total_cases": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total else 0.0,
        "invalid_route_outputs": invalid_count,
        "deterministic_or_smalltalk_shortcuts": deterministic_count,
        "llm_router_calls": llm_count,
        "router_llm_call_avoidance_rate": round(deterministic_count / total, 4) if total else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latency_ms), 2) if latency_ms else 0.0,
            "p50": round(percentile(latency_ms, 0.50), 2),
            "p95": round(percentile(latency_ms, 0.95), 2),
        },
        "confusion": {expected: dict(predicted) for expected, predicted in confusion.items()},
        "cases": results,
        "scope_note": (
            "Routing evaluation only. This does not measure retrieval Recall@k, answer faithfulness, "
            "citation coverage, tool correctness, or real token/cost usage."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
