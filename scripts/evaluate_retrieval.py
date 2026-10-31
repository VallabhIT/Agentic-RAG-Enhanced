"""Evaluate hybrid retrieval with Recall@k and MRR against a small source-level gold set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.hybrid_retriever import retrieve_hybrid_documents

ROOT = Path(__file__).resolve().parents[1]


def source_titles(docs):
    return [str((doc.metadata or {}).get("title", "")) for doc in docs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        docs = retrieve_hybrid_documents(case["query"], k=args.k)
        titles = source_titles(docs)
        expected = set(case["expected_sources"])
        hits = [index + 1 for index, title in enumerate(titles) if title in expected]
        recall = 1.0 if hits else 0.0
        mrr = 1.0 / hits[0] if hits else 0.0
        rows.append((case["id"], recall, mrr, titles))
    mean_recall = sum(row[1] for row in rows) / max(len(rows), 1)
    mean_mrr = sum(row[2] for row in rows) / max(len(rows), 1)
    print(f"Recall@{args.k}: {mean_recall:.3f}")
    print(f"MRR@{args.k}: {mean_mrr:.3f}")
    for case_id, recall, mrr, titles in rows:
        print(f"- {case_id}: recall={recall:.1f} mrr={mrr:.3f} top={titles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
