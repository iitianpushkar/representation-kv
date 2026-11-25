# Representation-guided Qwen KV retention

Inference-time KV compression for **Qwen2.5-7B-Instruct** on **LongBench v1**.
This is a batch-size-one research implementation pinned to Transformers 4.57.1.

## Method

For each token at each layer, score the cosine distance between the hidden
state entering that layer and the hidden state at the **last layer where this
same token was retained**. Layer 1 retains every token. After full causal
attention during prefill, each later layer keeps the top
`ceil(retention_ratio * prompt_tokens)` K/V pairs. Only selected tokens update
their comparison reference. The prompt's full K/V exist temporarily for each
layer's prefill attention; only selected K/V persist afterward.

During greedy decoding, a new token attends to retained historical K/V plus
its own current K/V at each layer. The online cache keeps at most
`ceil(retention_ratio * tokens_seen)` entries per later layer. If full, a new
entry replaces the retained entry with the lowest selection score only when
its score is higher. This is an online ranking policy: previously discarded
entries cannot re-enter when the budget grows. Layer 1 always stores all K/V.
All keys retain their original RoPE positions, and no layer reads another
layer's K/V. Hidden-state references are discarded after a token has passed
the final layer, so they are temporary rather than a persistent per-token
cache. No recent-window or sink-token protection is applied in this version.

## Rorqual setup

Use a networked login or transfer node to stage dependencies. The experiment
is launched by one Python script inside a GPU allocation; create your own
Rorqual job around that command. The existing EMO benchmark repo also stages
model/data before GPU runs and documents commands in each experiment folder.

```bash
git clone https://github.com/iitianpushkar/representation-kv.git representation-kv
cd representation-kv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
git clone https://github.com/THUDM/LongBench.git ../LongBench
.venv/bin/hf download Qwen/Qwen2.5-7B-Instruct --local-dir ../models/qwen2.5-7b-instruct
.venv/bin/python scripts/stage_longbench.py --data-dir ../datasets/longbench-v1
.venv/bin/python scripts/verify_small.py
```

The staging command extracts the five selected JSONL tasks and records the
official archive SHA-256 and row counts in `manifest.json`.

Install a PyTorch wheel compatible with Rorqual's CUDA version if the default
wheel is unsuitable. The [retention curve experiment guide](experiments/qwen2_5_7b_longbench_retention_curve/README.md)
contains the login-node preflight, H100 smoke run, and single full-run command.
The default curve runs **100%, 70%, 50%, 30%, 10%, 5%, 1%**, in that order.
`scripts/run_curve.py` loads Qwen once, resumes each condition independently,
and writes a curve summary after each completed ratio. There is no committed
Slurm job file.

## Outputs

Each task has a JSONL file with predictions, references, token counts,
prefill/decode time, KV bytes, metadata bytes, layer retention counts, and
peak allocated GPU bytes. `scores.json` uses the official LongBench v1 task
metrics from the staged checkout. The default tasks are Qasper, HotpotQA,
2WikiMultihopQA, GovReport, and PassageRetrieval-en.

Each prediction also records `layer_retention_prefill`,
`layer_retention_final`, `global_retention_prefill`, and
`global_retention_final`. The per-layer ratios divide retained entries by
tokens processed at that stage; the global ratios divide the sum of retained
entries by the original full-KV entry count across all layers. With the
current fixed budget, Layer 1 is 100% and all later layers target the same
ratio (up to rounding and online selection). For 28 layers, the approximate
global ratio is `(1 + 27 * later_layer_ratio) / 28`. Compare experiments at
their **measured global ratio**, particularly against methods with adaptive
budgets. Adaptive allocation across layers is not implemented in this version.

Compressed runs also save `scores/<task>/<index>.npz` for each example. Its
`prefill_scores`, `prefill_selected`, and `prefill_reference_layer` arrays
have shape `[layers, prompt_tokens]`, including scores for skipped tokens.
The corresponding `decode_*` arrays have shape `[layers, decoded_tokens]` and
include the score, selection decision, prior reference layer, and evicted
position for every processed generated token. Layer indices are zero-based;
`-1` means no prior layer, and Layer 1 scores are `NaN` because it is always
retained. The last generated token is not decoded if generation stops there,
so it has no decode trace column. `score_trace` in each JSONL row points to
its NPZ file. These trace files are saved before the prediction row to keep
resumed runs consistent.

The runner reserves each task's generation length within the model's 32K
position limit and truncates overly long prompts in the middle. It uses the
Qwen chat template, greedy generation, BF16, and one GPU. `--limit` and
`--max-new-tokens` are for smoke tests; do not compare their scores with full
LongBench results.

## Limitations

- Prefill attention FLOPs are unchanged; selection adds overhead.
- Decode cache updates use PyTorch concatenation and may limit wall-clock
  speedups even when fewer K/V entries are read. Compare measured latency.
- Only unpadded batch size 1, greedy decoding, Qwen2.5, and full causal
  attention are supported. Beam search and sliding-window variants are not.
- Before the curve, the runner checks that the custom 100% cache matches the
  unmodified Qwen cache on one real LongBench example.
