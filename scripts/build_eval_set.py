#!/usr/bin/env python3
"""Build an evaluation set from the hosted customer-support dataset.

Usage:
    python scripts/build_eval_set.py --limit 200 --out data/eval/generated.jsonl

Requires the ``datasets`` package and network access. The default dataset is
declared in ``policy_assistant.config.EVAL_DATASET``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_assistant.config import EVAL_DATASET  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=EVAL_DATASET)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/eval/generated.jsonl"))
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split="train")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        for row in dataset.select(range(min(args.limit, len(dataset)))):
            handle.write(
                json.dumps(
                    {
                        "question": row.get("instruction") or row.get("question"),
                        "reference": row.get("response") or row.get("answer"),
                        "category": row.get("category"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {min(args.limit, len(dataset))} rows to {args.out}")


if __name__ == "__main__":
    main()
