#!/usr/bin/env python3
"""Download selected LongBench v1 tasks once on a networked login node."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from representation_kv.longbench import DEFAULT_TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--hf-cache", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        target = data_dir / f"{task}.jsonl"
        if target.exists():
            print(f"Already staged: {target}")
            continue
        dataset = load_dataset("THUDM/LongBench", task, split="test", cache_dir=args.hf_cache)
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for item in dataset:
                stream.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        temporary.replace(target)
        print(f"Staged {len(dataset)} examples: {target}")


if __name__ == "__main__":
    main()
