"""LongBench v1 local staging, prompts and official metric integration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import torch


DEFAULT_TASKS = (
    "qasper", "hotpotqa", "2wikimqa", "gov_report", "passage_retrieval_en"
)


def official_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "LongBench" / "config" / "dataset2prompt.json").exists():
        return root / "LongBench"
    if (root / "config" / "dataset2prompt.json").exists():
        return root
    raise FileNotFoundError(f"Could not find LongBench/config under {root}")


def task_configs(root: str | Path) -> tuple[dict, dict]:
    root = official_root(root)
    prompts = json.loads((root / "config" / "dataset2prompt.json").read_text())
    max_lengths = json.loads((root / "config" / "dataset2maxlen.json").read_text())
    return prompts, max_lengths


def records(data_dir: str | Path, task: str):
    source = Path(data_dir) / f"{task}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"Stage {task} first: {source}")
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            yield index, json.loads(line)


def encode_prompt(tokenizer, formatted: str, max_prompt_tokens: int):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
    )
    original_tokens = ids.shape[1]
    if ids.shape[1] > max_prompt_tokens:
        left = (max_prompt_tokens + 1) // 2
        right = max_prompt_tokens - left
        ids = ids[:, :left] if right == 0 else torch.cat(
            (ids[:, :left], ids[:, -right:]), dim=1
        )
    return ids, original_tokens


def official_metric(root: str | Path, task: str):
    root = official_root(root)
    sys.path.insert(0, str(root))
    from metrics import qa_f1_score, retrieval_score, rouge_score  # type: ignore

    return {
        "qasper": qa_f1_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "gov_report": rouge_score,
        "passage_retrieval_en": retrieval_score,
    }[task]
