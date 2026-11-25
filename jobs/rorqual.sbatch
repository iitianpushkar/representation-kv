#!/usr/bin/env bash
#SBATCH --job-name=representation-kv
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Supply the Rorqual account/partition at submission time if required there.
# Required exports: REPO_DIR, VENV_PYTHON, MODEL_PATH, LONGBENCH_CODE,
# DATA_DIR, OUTPUT_DIR, MODE. RATIO defaults to 1.0.
: "${REPO_DIR:?Set REPO_DIR to this repository checkout}"
: "${VENV_PYTHON:?Set VENV_PYTHON to the Python in the experiment environment}"
: "${MODEL_PATH:?Set MODEL_PATH to the staged Qwen model directory}"
: "${LONGBENCH_CODE:?Set LONGBENCH_CODE to the official LongBench checkout}"
: "${DATA_DIR:?Set DATA_DIR to staged LongBench task JSONL files}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique result directory}"
: "${MODE:?Set MODE to full or compressed}"
RATIO="${RATIO:-1.0}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO_DIR"

"$VENV_PYTHON" -u scripts/run_longbench.py \
  --model-path "$MODEL_PATH" \
  --longbench-code "$LONGBENCH_CODE" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --ratio "$RATIO" \
  ${EXTRA_ARGS:-}

"$VENV_PYTHON" -u scripts/score_longbench.py \
  --longbench-code "$LONGBENCH_CODE" \
  --output-dir "$OUTPUT_DIR"
