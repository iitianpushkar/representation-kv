#!/usr/bin/env python3
"""Apply the official LongBench v1 metrics to completed prediction files."""

import argparse
import json
from pathlib import Path

from representation_kv.longbench import DEFAULT_TASKS, official_metric


def score_condition(longbench_code: str, output_dir: str | Path, tasks):
    output = Path(output_dir)
    scores = {}
    for task in tasks:
        path = output / f"{task}.jsonl"
        if not path.exists():
            continue
        metric = official_metric(longbench_code, task)
        values = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                values.append(max(metric(row["pred"], answer, all_classes=row["all_classes"])
                                  for answer in row["answers"]))
        if values:
            scores[task] = round(100 * sum(values) / len(values), 2)
            print(f"{task}: {scores[task]} ({len(values)} examples)")
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--longbench-code", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    args = parser.parse_args()
    score_condition(args.longbench_code, args.output_dir, args.tasks)


if __name__ == "__main__":
    main()
