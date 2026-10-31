"""DeepEval suite for the RAG subgraph.

Metrics intentionally cover both retriever and generator quality:
- Contextual Precision: are useful chunks ranked early?
- Contextual Recall: did retrieval contain the evidence needed by the reference answer?
- Contextual Relevancy: how much retrieved context is relevant to the query?
- Faithfulness: are answer claims supported by retrieved context?
- Answer Relevancy: does the answer address the question?
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app import config
from app.hybrid_retriever import format_retrieved_documents, retrieve_hybrid_documents
from app.prompts import FINALIZE_PROMPT
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parents[1]


def generate_answer(query: str, docs: list[str]) -> str:
    context = "\n---\n".join(f"[Doc {i + 1}] {doc}" for i, doc in enumerate(docs))
    prompt = FINALIZE_PROMPT.format(
        original_query=query,
        rewritten_query=query,
        source_note="Answer from the internal knowledge base.",
        tool_context=f"## Internal knowledge-base documents\n{context}",
    )
    return str(config.get_llm().invoke(prompt).content).strip()


def score_metric(metric, test_case):
    try:
        metric.measure(test_case)
        return float(metric.score or 0.0), str(metric.reason or "")
    except Exception as exc:
        return None, f"evaluation error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Evaluate only one case id")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evals/rag_cases.json").read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
    if not cases:
        raise SystemExit("No matching RAG cases")

    model = os.getenv("DEEPEVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    metric_factories = [
        ("contextual_precision", lambda: ContextualPrecisionMetric(threshold=0.7, model=model, include_reason=True)),
        ("contextual_recall", lambda: ContextualRecallMetric(threshold=0.7, model=model, include_reason=True)),
        ("contextual_relevancy", lambda: ContextualRelevancyMetric(threshold=0.7, model=model, include_reason=True)),
        ("faithfulness", lambda: FaithfulnessMetric(threshold=0.7, model=model, include_reason=True)),
        ("answer_relevancy", lambda: AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True)),
    ]
    totals = {name: [] for name, _ in metric_factories}
    for case in cases:
        docs = format_retrieved_documents(retrieve_hybrid_documents(case["input"], k=config.RAG_TOP_K))
        answer = generate_answer(case["input"], docs)
        test_case = LLMTestCase(
            name=case["id"],
            input=case["input"],
            actual_output=answer,
            expected_output=case["expected_output"],
            retrieval_context=docs,
        )
        print(f"\n[{case['id']}]\nAnswer: {answer}")
        for name, factory in metric_factories:
            score, reason = score_metric(factory(), test_case)
            if score is not None:
                totals[name].append(score)
                print(f"  {name}: {score:.3f} — {reason}")
            else:
                print(f"  {name}: ERROR — {reason}")

    print("\nMean scores")
    for name, values in totals.items():
        print(f"{name}: {sum(values)/len(values):.3f}" if values else f"{name}: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
