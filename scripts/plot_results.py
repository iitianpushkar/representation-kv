#!/usr/bin/env python3
"""Render the LongBench retention curve summaries as PNG charts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from representation_kv.progress import configure_logging, log


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_LABELS = {
    "qasper": "Qasper",
    "hotpotqa": "HotpotQA",
    "2wikimqa": "2WikiMultihopQA",
    "gov_report": "GovReport",
    "passage_retrieval_en": "PassageRetrieval-en",
}
TASK_METRICS = {
    "qasper": "F1",
    "hotpotqa": "F1",
    "2wikimqa": "F1",
    "gov_report": "ROUGE-L",
    "passage_retrieval_en": "Accuracy",
}
CHARTS = (
    ("quality_vs_retention.png", "global_retention_final", "longbench_score",
     "LongBench quality vs KV retention", "Measured global KV retention (%)", "LongBench score"),
    ("quality_vs_kv_memory.png", "kv_mib_final", "longbench_score",
     "LongBench quality vs KV memory", "Mean final KV cache (MiB)", "LongBench score"),
    ("kv_memory_vs_retention.png", "global_retention_final", "kv_mib_final",
     "KV memory vs retention", "Measured global KV retention (%)", "Mean final KV cache (MiB)"),
    ("decode_time_vs_retention.png", "global_retention_final", "decode_seconds",
     "Decode time vs KV retention", "Measured global KV retention (%)",
     "Mean decode time per example (s)"),
    ("prefill_time_vs_retention.png", "global_retention_final", "prefill_seconds",
     "Prefill time vs KV retention", "Measured global KV retention (%)",
     "Mean prefill time per example (s)"),
)


def plot_summary(output_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    summary = root / "curve_summary.json"
    rows = json.loads(summary.read_text())
    if not rows:
        log.info("PLOTS_SKIPPED reason=empty_summary path=%s", summary)
        return []
    created = []
    for filename, x_key, y_key, title, xlabel, ylabel in CHARTS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        plotted = False
        for task, label in TASK_LABELS.items():
            points = []
            for row in rows:
                if row["task"] != task or row.get(x_key) is None or row.get(y_key) is None:
                    continue
                x = float(row[x_key])
                y = float(row[y_key])
                if math.isfinite(x) and math.isfinite(y):
                    points.append((x * 100 if x_key == "global_retention_final" else x, y))
            if not points:
                continue
            points.sort()
            ax.plot([x for x, _ in points], [y for _, y in points],
                    marker="o", linewidth=1.8, markersize=5, label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if x_key == "global_retention_final":
            ax.set_xlim(0, 105)
        if y_key == "longbench_score":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = root / filename
        temporary = root / f".{filename}"
        fig.savefig(temporary, dpi=160)
        plt.close(fig)
        temporary.replace(path)
        created.append(path)
        log.info("PLOT_WRITTEN path=%s", path)
    conditions = sorted(
        {row["condition"] for row in rows if row.get("longbench_score") is not None},
        key=lambda name: int(name[1:]), reverse=True,
    )
    tasks = [task for task in TASK_LABELS if any(
        row["task"] == task and row.get("longbench_score") is not None for row in rows
    )]
    if conditions and tasks:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        centers = list(range(len(conditions)))
        width = 0.8 / len(tasks)
        scores = {
            (row["condition"], row["task"]): float(row["longbench_score"])
            for row in rows if row.get("longbench_score") is not None
        }
        for task_index, task in enumerate(tasks):
            positions = []
            values = []
            for condition_index, condition in enumerate(conditions):
                value = scores.get((condition, task))
                if value is not None:
                    positions.append(condition_index + (task_index - (len(tasks) - 1) / 2) * width)
                    values.append(value)
            ax.bar(positions, values, width=width,
                   label=f"{TASK_LABELS[task]} ({TASK_METRICS[task]})")
        ax.set(
            title="LongBench task metrics by requested KV retention",
            xlabel="Retention in layers after Layer 1",
            ylabel="LongBench score (0–100)",
        )
        ax.set_xticks(centers, [f"{int(name[1:])}%" for name in conditions])
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        fig.tight_layout()
        path = root / "metric_comparison_by_ratio.png"
        temporary = root / ".metric_comparison_by_ratio.png"
        fig.savefig(temporary, dpi=160, bbox_inches="tight")
        plt.close(fig)
        temporary.replace(path)
        created.append(path)
        log.info("PLOT_WRITTEN path=%s", path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(
        REPO_ROOT / "results/qwen2_5_7b_longbench_retention_curve"))
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    configure_logging(root / "run.log")
    plot_summary(root)


if __name__ == "__main__":
    main()
