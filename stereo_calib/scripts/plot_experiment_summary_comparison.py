#!/usr/bin/env python3
"""Plot side-by-side comparison for two experiment summary JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_scene_name(scene: str) -> str:
    scene = re.sub(r"_no_margin$", "", scene)
    match = re.search(r"drive_(\d+)_sync", scene)
    if match:
        return match.group(1)
    return scene


def index_runs(summary: dict) -> dict[str, dict]:
    indexed = {}
    for run in summary.get("runs", []):
        indexed[normalize_scene_name(str(run.get("scene", run.get("label", ""))))] = run
    return indexed


def bar_panel(ax, labels, values_a, values_b, name_a, name_b, title, ylabel, value_fmt):
    x = np.arange(len(labels))
    width = 0.36
    bars_a = ax.bar(x - width / 2, values_a, width, label=name_a, color="#4c78a8")
    bars_b = ax.bar(x + width / 2, values_b, width, label=name_b, color="#f58518")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    top = max([0, *values_a, *values_b])
    offset = top * 0.025 if top else 0.1
    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                value_fmt(height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment summary JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline summary JSON, e.g. margin.")
    parser.add_argument("--candidate", required=True, help="Candidate summary JSON, e.g. no_margin.")
    parser.add_argument("--baseline_label", default="margin", help="Label for baseline bars.")
    parser.add_argument("--candidate_label", default="no_margin", help="Label for candidate bars.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = load_summary(baseline_path)
    candidate = load_summary(candidate_path)

    baseline_runs = index_runs(baseline)
    candidate_runs = index_runs(candidate)
    labels = [label for label in baseline_runs if label in candidate_runs]
    if not labels:
        raise ValueError("No matching scenes found between the two summaries.")

    time_a = [baseline_runs[label]["time"]["seconds"] for label in labels]
    time_b = [candidate_runs[label]["time"]["seconds"] for label in labels]
    matches_a = [baseline_runs[label]["matches"]["total_points"] for label in labels]
    matches_b = [candidate_runs[label]["matches"]["total_points"] for label in labels]
    obs_a = [baseline_runs[label]["ba"]["observations"] for label in labels]
    obs_b = [candidate_runs[label]["ba"]["observations"] for label in labels]

    total_a = baseline.get("total", {})
    total_b = candidate.get("total", {})

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f"KITTI City Small Test: {args.baseline_label} vs {args.candidate_label}\n"
        f"{baseline_path.name}  vs  {candidate_path.name}",
        fontsize=13,
        fontweight="bold",
    )

    bar_panel(
        axes[0, 0],
        labels,
        time_a,
        time_b,
        args.baseline_label,
        args.candidate_label,
        "Total Runtime per Scene",
        "seconds",
        lambda v: f"{v:.1f}s",
    )
    bar_panel(
        axes[0, 1],
        labels,
        matches_a,
        matches_b,
        args.baseline_label,
        args.candidate_label,
        "Matcher Output Match Points",
        "points",
        lambda v: f"{int(v):,}",
    )
    bar_panel(
        axes[1, 0],
        labels,
        obs_a,
        obs_b,
        args.baseline_label,
        args.candidate_label,
        "Final BA Observations",
        "observations",
        lambda v: f"{int(v):,}",
    )

    axes[1, 1].axis("off")
    rows = [
        ["metric", args.baseline_label, args.candidate_label, "candidate / baseline"],
        [
            "total time",
            f"{total_a.get('seconds', 0):.1f}s",
            f"{total_b.get('seconds', 0):.1f}s",
            f"{total_b.get('seconds', 0) / max(total_a.get('seconds', 1), 1):.3f}x",
        ],
        [
            "match points",
            f"{int(total_a.get('match_points', 0)):,}",
            f"{int(total_b.get('match_points', 0)):,}",
            f"{total_b.get('match_points', 0) / max(total_a.get('match_points', 1), 1):.3f}x",
        ],
        [
            "BA observations",
            f"{sum(obs_a):,}",
            f"{sum(obs_b):,}",
            f"{sum(obs_b) / max(sum(obs_a), 1):.3f}x",
        ],
    ]
    table = axes[1, 1].table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        else:
            cell.set_facecolor("#f8f8f8")

    axes[0, 0].legend(loc="upper right")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = baseline_path.with_name(
            f"{baseline_path.stem}_vs_{candidate_path.stem}_comparison.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


if __name__ == "__main__":
    main()
