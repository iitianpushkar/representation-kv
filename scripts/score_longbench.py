#!/usr/bin/env python3
"""Apply the official LongBench v1 metrics to completed prediction files."""

import argparse
import json
from pathlib import Path
import time

from representation_kv.longbench import DEFAULT_TASKS, official_metric
from representation_kv.progress import configure_logging, log

REPO_ROOT = Path(__file__).resolve().parents[1]


def score_condition(longbench_code: str, output_dir: str | Path, tasks):
    output = Path(output_dir)
    configure_logging(output / "run.log")
    scores = {}
    for task in tasks:
        path = output / f"{task}.jsonl"
        if not path.exists():
            log.info("SCORING_SKIP task=%s reason=no_predictions", task)
            continue
        task_start = time.perf_counter()
        log.info("SCORING_TASK_START task=%s path=%s", task, path)
        metric = official_metric(longbench_code, task)
        values = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                values.append(max(metric(row["pred"], answer, all_classes=row["all_classes"])
                                  for answer in row["answers"]))
        if values:
            scores[task] = round(100 * sum(values) / len(values), 2)
            log.info("SCORING_TASK_DONE task=%s score=%.2f examples=%d seconds=%.2f",
                     task, scores[task], len(values), time.perf_counter() - task_start)
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    log.info("SCORING_DONE tasks=%d path=%s", len(scores), output / "scores.json")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--longbench-code", default=str(REPO_ROOT / "third_party/LongBench"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    args = parser.parse_args()
    score_condition(args.longbench_code, args.output_dir, args.tasks)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("SCORING_FAILED")
        raise
