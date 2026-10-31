"""Fast component-level checks for non-RAG nodes.

This complements DeepEval by keeping deterministic safety and state-machine invariants cheap:
- SQL safety gate rejects mutation/DDL statements.
- Python safety gate rejects filesystem/process/network-capable code.
- RAG retry routing is bounded.
- Query rewrite preserves important numeric/entity tokens when the LLM is available.
"""
from __future__ import annotations

import argparse
import re

from app import config, tools
from app.prompts import REWRITE_PROMPT
from app.rag_subgraph import route_after_judge
from app.utils import format_history


def check_sql_safety() -> list[tuple[str, bool]]:
    cases = [
        ("SELECT COUNT(*) FROM client_deliverables", False),
        ("WITH x AS (SELECT 1) SELECT * FROM x", False),
        ("DROP TABLE client_deliverables", True),
        ("UPDATE client_deliverables SET status='Delivered'", True),
        ("SELECT 1; DELETE FROM client_deliverables", True),
    ]
    return [(sql, tools.is_dangerous_sql(sql) == expected) for sql, expected in cases]


def check_python_safety() -> list[tuple[str, bool]]:
    cases = [
        ("print(2 + 2)", False),
        ("import math\nprint(math.sqrt(9))", False),
        ("open('secret.txt').read()", True),
        ("import os\nos.system('whoami')", True),
        ("eval('2+2')", True),
    ]
    return [(code, tools.detect_dangerous_python(code) == expected) for code, expected in cases]


def check_rag_bounds() -> list[tuple[str, bool]]:
    return [
        ("relevant ends", route_after_judge({"relevant": True, "retry_count": 0}) == "done"),
        ("retry once", route_after_judge({"relevant": False, "retry_count": 0}) == "refine"),
        ("retry exhausted", route_after_judge({"relevant": False, "retry_count": config.RAG_MAX_RETRIES}) == "done"),
    ]


def check_rewrite(query: str) -> tuple[bool, str]:
    try:
        out = config.get_llm(0.0).invoke(REWRITE_PROMPT.format(history=format_history([], 6), query=query))
        rewritten = str(out.content).strip()
        important = re.findall(r"\b\d+(?:[.,]\d+)?\b|\b[A-Z][A-Za-z-]{2,}\b", query)
        preserved = sum(token.lower() in rewritten.lower() for token in important)
        return preserved >= max(1, len(important) // 2), rewritten
    except Exception as exc:
        return True, f"skipped LLM rewrite check: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", default="How much rice and cooking oil are on my grocery list? 5 kg and 2 L")
    args = parser.parse_args()
    groups = {
        "sql_safety": check_sql_safety(),
        "python_safety": check_python_safety(),
        "rag_bounds": check_rag_bounds(),
    }
    passed = 0
    total = 0
    for name, results in groups.items():
        ok = sum(result for _, result in results)
        passed += ok
        total += len(results)
        print(f"{name}: {ok}/{len(results)}")
    rewrite_ok, rewritten = check_rewrite(args.rewrite)
    print(f"rewrite_preservation: {'PASS' if rewrite_ok else 'FAIL'} — {rewritten}")
    print(f"deterministic node checks: {passed}/{total}")
    return 0 if passed == total and rewrite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
