#!/usr/bin/env python3
"""One Qwen2.5/LongBench retention curve, 100% down to 1%."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import torch

from representation_kv.longbench import (
    DEFAULT_TASKS, encode_prompt, official_root, records, task_configs,
)
from representation_kv.qwen import CompressedQwen, generate_full
from representation_kv.progress import configure_logging, log
from plot_results import plot_summary
from run_longbench import load_model_and_tokenizer, run_condition
from score_longbench import score_condition


DEFAULT_RATIOS = (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
REPO_ROOT = Path(__file__).resolve().parents[1]


def ratio_name(ratio: float) -> str:
    percent = round(ratio * 100)
    if abs(ratio * 100 - percent) > 1e-8 or not 1 <= percent <= 100:
        raise ValueError("Ratios must be whole percentages from 1 to 100")
    return f"r{percent:03d}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models/qwen2.5-7b-instruct"))
    parser.add_argument("--longbench-code", default=str(REPO_ROOT / "third_party/LongBench"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/longbench-v1"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results/qwen2_5_7b_longbench_retention_curve"),
                        help="Directory for the curve; reruns resume completed examples")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def verify_inputs(args):
    ratios = list(args.ratios)
    names = [ratio_name(ratio) for ratio in ratios]
    if len(names) != len(set(names)) or ratios != sorted(ratios, reverse=True):
        raise ValueError("Ratios must be unique and in descending order")
    if not ratios or ratios[0] != 1.0:
        raise ValueError("The first condition must be 100% retention")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_context < 2:
        raise ValueError("--max-context must be at least 2")
    model_path = Path(args.model_path).expanduser().resolve()
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing model config: {config_file}")
    model_config = json.loads(config_file.read_text())
    if model_config.get("model_type") != "qwen2" or model_config.get("num_hidden_layers") != 28:
        raise ValueError("Expected the Qwen2.5-7B-Instruct checkpoint")
    if not (model_path / "model.safetensors.index.json").exists() and not (
        model_path / "model.safetensors"
    ).exists():
        raise FileNotFoundError(f"Missing model safetensors under {model_path}")
    root = official_root(args.longbench_code)
    prompts = json.loads((root / "config" / "dataset2prompt.json").read_text())
    maximums = json.loads((root / "config" / "dataset2maxlen.json").read_text())
    data_dir = Path(args.data_dir).expanduser().resolve()
    for task in args.tasks:
        if task not in prompts or task not in maximums:
            raise ValueError(f"Missing official LongBench config for {task}")
        if args.max_context <= (args.max_new_tokens or maximums[task]):
            raise ValueError(f"max-context leaves no prompt tokens for {task}")
        source = data_dir / f"{task}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing staged LongBench task: {source}")
        with source.open(encoding="utf-8") as stream:
            first = next((json.loads(line) for line in stream if line.strip()), None)
        if first is None or not {"context", "input", "answers", "all_classes"} <= first.keys():
            raise ValueError(f"Invalid LongBench task file: {source}")
        log.info("INPUT_READY task=%s data=%s max_new_tokens=%d", task, source,
                 args.max_new_tokens or maximums[task])
    log.info("PREFLIGHT_DONE model=%s ratios=%s tasks=%s", model_path,
             ",".join(names), ",".join(args.tasks))
    return names


def write_summary(root: Path, names: list[str], tasks: list[str]):
    rows = []
    truncations = []
    for name in names:
        condition = root / name
        scores_path = condition / "scores.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        for task in tasks:
            path = condition / f"{task}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as stream:
                examples = [json.loads(line) for line in stream if line.strip()]
            if not examples:
                continue
            mean = lambda key: sum(float(row[key]) for row in examples) / len(examples)
            for example in examples:
                removed = int(example.get("removed_prompt_tokens", 0))
                if removed:
                    truncations.append({
                        "condition": name, "task": task, "index": example["index"],
                        "original_prompt_tokens": example["original_prompt_tokens"],
                        "prompt_token_limit": example.get("prompt_token_limit"),
                        "prompt_tokens": example["prompt_tokens"],
                        "removed_prompt_tokens": removed,
                    })
            rows.append({
                "condition": name,
                "requested_later_layer_retention": int(name[1:]) / 100,
                "task": task,
                "examples": len(examples),
                "truncated_examples": sum(bool(row.get("removed_prompt_tokens", 0))
                                          for row in examples),
                "longbench_score": scores.get(task),
                "global_retention_prefill": mean("global_retention_prefill"),
                "global_retention_final": mean("global_retention_final"),
                "kv_mib_prefill": mean("kv_bytes_prefill") / 2**20,
                "kv_mib_final": mean("kv_bytes_final") / 2**20,
                "prefill_seconds": mean("prefill_seconds"),
                "decode_seconds": mean("decode_seconds"),
                "generated_tokens": mean("generated_tokens"),
            })
    if rows:
        with (root / "curve_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (root / "truncation_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "condition", "task", "index", "original_prompt_tokens",
            "prompt_token_limit", "prompt_tokens", "removed_prompt_tokens",
        ])
        writer.writeheader()
        writer.writerows(truncations)
    (root / "curve_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    log.info("SUMMARY_WRITTEN rows=%d truncations=%d path=%s", len(rows),
             len(truncations), root / "curve_summary.csv")


def check_full_baseline(args, root: Path, model, tokenizer):
    """Gate the sweep on a real-model 100%-retention token match."""
    report_path = root / "baseline_check.json"
    task = args.tasks[0]
    _, maximums = task_configs(args.longbench_code)
    index, item = next(records(args.data_dir, task))
    max_context = min(args.max_context, model.config.max_position_embeddings)
    prompt_limit = max_context - (args.max_new_tokens or maximums[task])
    ids, original_prompt_tokens = encode_prompt(tokenizer, json.loads(
        (official_root(args.longbench_code) / "config" / "dataset2prompt.json").read_text()
    )[task].format(**item), prompt_limit)
    ids = ids.to("cuda:0")
    check_tokens = min(16, args.max_new_tokens or maximums[task])
    log.info("BASELINE_START task=%s index=%d prompt_tokens=%d compare_tokens=%d",
             task, index, ids.shape[1], check_tokens)
    baseline_start = time.perf_counter()
    full = generate_full(model, ids, check_tokens, tokenizer.eos_token_id)
    compressed = CompressedQwen(model, 1.0).generate(
        ids, check_tokens, tokenizer.eos_token_id
    )
    matched = full.token_ids == compressed.token_ids
    report = {
        "task": task, "index": index, "prompt_tokens": ids.shape[1],
        "original_prompt_tokens": original_prompt_tokens,
        "removed_prompt_tokens": original_prompt_tokens - ids.shape[1],
        "compared_generated_tokens": len(full.token_ids), "matched": matched,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not matched:
        raise RuntimeError(f"100% cache output differs from full Qwen: {report_path}")
    log.info("BASELINE_DONE matched=true task=%s index=%d seconds=%.2f report=%s",
             task, index, time.perf_counter() - baseline_start, report_path)


def main():
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    configure_logging(None if args.verify_only else output / "run.log")
    run_start = time.perf_counter()
    log.info("RUN_START verify_only=%s output=%s", args.verify_only, args.output_dir)
    log.info("PREFLIGHT_START")
    names = verify_inputs(args)
    if args.verify_only:
        log.info("VERIFY_ONLY_DONE no_gpu_model_loaded=true no_results_written=true")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The curve requires one CUDA GPU")
    output.mkdir(parents=True, exist_ok=True)
    log.info("RESULTS_READY output=%s log=%s", output, output / "run.log")
    log.info("MODEL_LOAD_START path=%s", args.model_path)
    model_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    log.info("MODEL_LOAD_DONE seconds=%.2f device=cuda:0", time.perf_counter() - model_start)
    check_full_baseline(args, output, model, tokenizer)
    for condition_index, (ratio, name) in enumerate(zip(args.ratios, names), start=1):
        condition_start = time.perf_counter()
        log.info("CONDITION_START name=%s step=%d/%d retention=%.0f%%", name,
                 condition_index, len(names), ratio * 100)
        args.mode = "compressed"
        args.ratio = ratio
        args.output_dir = str(output / name)
        run_condition(args, model=model, tokenizer=tokenizer)
        log.info("SCORING_START condition=%s", name)
        score_condition(args.longbench_code, args.output_dir, args.tasks)
        write_summary(output, names, args.tasks)
        try:
            plot_summary(output)
        except Exception:
            log.exception("PLOTS_FAILED condition=%s summary=%s", name,
                          output / "curve_summary.json")
        log.info("CONDITION_DONE name=%s seconds=%.2f", name,
                 time.perf_counter() - condition_start)
    log.info("RUN_DONE seconds=%.2f summary=%s", time.perf_counter() - run_start,
             output / "curve_summary.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("RUN_FAILED")
        raise
