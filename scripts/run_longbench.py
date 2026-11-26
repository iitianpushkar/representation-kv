#!/usr/bin/env python3
"""Qwen2.5-7B LongBench v1 baseline/compression experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from representation_kv.longbench import DEFAULT_TASKS, encode_prompt, records, task_configs
from representation_kv.qwen import CompressedQwen, generate_full

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models/qwen2.5-7b-instruct"), help="Local Qwen2.5-7B-Instruct directory")
    parser.add_argument("--longbench-code", default=str(REPO_ROOT / "third_party/LongBench"), help="Vendored LongBench v1 files")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/longbench-v1"), help="Task JSONL directory")
    parser.add_argument("--output-dir", required=True, help="Directory for this condition; existing examples are skipped")
    parser.add_argument("--mode", choices=("full", "compressed"), required=True)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override LongBench task default; useful for smoke tests")
    parser.add_argument("--limit", type=int, default=None, help="First N examples per task")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen2ForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(torch.device("cuda:0")).eval()
    if model.config.model_type != "qwen2" or model.config.num_hidden_layers != 28:
        raise ValueError("Expected Qwen2.5-7B-Instruct (28 Qwen2 layers)")
    return model, tokenizer


def run_condition(args, model=None, tokenizer=None):
    if not torch.cuda.is_available():
        raise RuntimeError("The LongBench runner requires one CUDA GPU")
    if args.mode == "compressed" and not 0 < args.ratio <= 1:
        raise ValueError("--ratio must be in (0,1]")
    if args.mode == "full" and args.ratio != 1.0:
        raise ValueError("The full baseline must use --ratio 1.0")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    prompts, max_lengths = task_configs(args.longbench_code)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
    max_context = min(args.max_context, model.config.max_position_embeddings)
    compressor = CompressedQwen(model, args.ratio) if args.mode == "compressed" else None

    for task in args.tasks:
        max_new = args.max_new_tokens or max_lengths[task]
        max_prompt = max_context - max_new
        if max_prompt < 1:
            raise ValueError("max-context must exceed the task generation length")
        target = output / f"{task}.jsonl"
        completed = set()
        if target.exists():
            with target.open(encoding="utf-8") as stream:
                completed = {json.loads(line)["index"] for line in stream if line.strip()}
            if compressor is not None:
                missing = [i for i in completed if not (output / "scores" / task / f"{i:06d}.npz").exists()]
                if missing:
                    raise RuntimeError(f"Missing score traces for completed {task} rows: {missing[:10]}")
        with target.open("a", encoding="utf-8") as stream:
            for index, item in records(args.data_dir, task):
                if args.limit is not None and index >= args.limit:
                    break
                if index in completed:
                    continue
                prompt = prompts[task].format(**item)
                input_ids, original_prompt_tokens = encode_prompt(tokenizer, prompt, max_prompt)
                input_ids = input_ids.to(device)
                removed_prompt_tokens = original_prompt_tokens - input_ids.shape[1]
                if removed_prompt_tokens:
                    print(f"TRUNCATED {task} {index}: {original_prompt_tokens} -> "
                          f"{input_ids.shape[1]} prompt tokens "
                          f"({removed_prompt_tokens} removed)", flush=True)
                if compressor is None:
                    result = generate_full(model, input_ids, max_new, tokenizer.eos_token_id)
                else:
                    result = compressor.generate(input_ids, max_new, tokenizer.eos_token_id)
                score_trace = None
                if compressor is not None:
                    score_trace = f"scores/{task}/{index:06d}.npz"
                    compressor.cache.save_trace(output / score_trace)
                prediction = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                prefill_ratios = [count / input_ids.shape[1]
                                  for count in result.layer_counts_prefill]
                processed_tokens = result.layer_counts_final[0]
                final_ratios = [count / processed_tokens
                                for count in result.layer_counts_final]
                row = {
                    "index": index, "pred": prediction, "answers": item["answers"],
                    "all_classes": item["all_classes"], "length": item.get("length"),
                    "prompt_tokens": input_ids.shape[1],
                    "original_prompt_tokens": original_prompt_tokens,
                    "prompt_token_limit": max_prompt,
                    "removed_prompt_tokens": removed_prompt_tokens,
                    "prompt_truncated": bool(removed_prompt_tokens),
                    "generated_tokens": len(result.token_ids),
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "kv_bytes_prefill": result.kv_bytes_prefill,
                    "kv_bytes_final": result.kv_bytes_final,
                    "metadata_bytes_final": result.metadata_bytes_final,
                    "layer_counts_prefill": result.layer_counts_prefill,
                    "layer_counts_final": result.layer_counts_final,
                    "layer_retention_prefill": prefill_ratios,
                    "layer_retention_final": final_ratios,
                    "global_retention_prefill": sum(result.layer_counts_prefill)
                    / (len(result.layer_counts_prefill) * input_ids.shape[1]),
                    "global_retention_final": sum(result.layer_counts_final)
                    / (len(result.layer_counts_final) * processed_tokens),
                    "peak_gpu_bytes": result.peak_gpu_bytes,
                    "score_trace": score_trace,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{task} {index}: prompt={input_ids.shape[1]} generated={len(result.token_ids)} "
                      f"KV={result.kv_bytes_final / 2**20:.1f} MiB", flush=True)


def main():
    run_condition(parse_args())


if __name__ == "__main__":
    main()
