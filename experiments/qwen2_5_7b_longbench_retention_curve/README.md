# Qwen2.5-7B LongBench representation-change retention curve

This is a **planned, unrun** experiment. It tests requested retention ratios
of **100%, 70%, 50%, 30%, 20%, 10%, 5%, and 1%**, in that execution order. The
100% point uses the same custom attention and cache path without dropping any
entries. Ratios apply to every layer after Layer 1; Layer 1 always keeps all
tokens. The current selector does not use a global adaptive budget. We save
both measured global and per-layer retention so that later budget experiments
can use the token score traces from this run.

## Fixed protocol

- Model: local `Qwen/Qwen2.5-7B-Instruct`, BF16, one GPU, batch size 1.
- Data: LongBench v1, with Qasper, HotpotQA, 2WikiMultihopQA, GovReport, and
  PassageRetrieval-en by default. Prompts, generation lengths, and scoring
  metrics come from the vendored official LongBench files in this repository.
- Prompt: Qwen chat template, middle truncation to leave room for generation
  within the model's 32K position limit. Greedy decoding, no sampling.
- Selection: cosine distance to the same token's representation at its last
  retained layer. Prefill attends over the entire prompt before each layer's
  top-K cache compaction. Decoding attends to that layer's retained cache and
  the current token's own K/V, then applies the same score and online budget.
- Output: each ratio has separate resumable predictions and per-example NPZ
  score traces. Traces contain every prompt token's score at every layer,
  whether it was selected, its previous retained layer, and the analogous
  fields for processed generated tokens. A run-level CSV/JSON summarizes
  LongBench score, measured global retention, KV size, and latency per task.

## Cluster commands

From the repository checkout on a networked login node, download the model as
described in the [root README](../../README.md#rorqual-setup). The LongBench
files are already present. Check the inputs without loading a GPU model or
writing results:

```bash
.venv/bin/python scripts/run_curve.py \
  --verify-only
```

Inside an H100 allocation, run a two-example smoke curve in a **separate**
output directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python -u scripts/run_curve.py \
  --output-dir results/qwen2_5_7b_longbench_smoke \
  --tasks hotpotqa --limit 2 --max-context 4096 --max-new-tokens 16
```

Then put this **one command** in your H100 job for the full curve:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python -u scripts/run_curve.py
```

The runner loads Qwen once and completes `r100`, `r070`, `r050`, `r030`,
`r020`, `r010`, `r005`, then `r001`. Running the same command again resumes in
the same output directory and skips completed examples.
Each ratio directory contains one prediction JSONL per task,
`scores/<task>/<index>.npz`, and `scores.json`. The run root contains
`baseline_check.json`, `curve_summary.csv`, and `curve_summary.json`.
`truncation_report.csv` lists every prompt shortened to fit the model context,
and the curve summary counts affected examples for each task and ratio.
No saved configuration is used to block a rerun. Before each GPU sweep, the script
compares up to 16 greedy tokens from the unmodified Qwen cache and the custom
100% cache on the first LongBench example; it stops if they differ.
Use the same model and settings when rerunning; the script does not compare
them with the earlier run.

The smoke run's shortened prompts and outputs are only for validating the
pipeline. Do not compare its scores with the full curve. The full Qwen7B
LongBench curve and H100 latency have not yet been measured.
