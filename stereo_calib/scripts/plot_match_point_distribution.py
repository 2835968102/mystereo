#!/usr/bin/env python3
"""Plot spatial distribution of matched points from a matches JSON file.

This is aimed at KITTI raw sequence matches, where each pair can be stereo
(image_00 -> image_01) or temporal (image_00 -> image_00 / image_01 -> image_01).
The script aggregates all match observations by actual camera side, then draws
scatter plots and grid-count heatmaps so sparse or biased regions are obvious.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pair_type(pair: dict) -> str:
    """Return LR / LL / RR / other using KITTI raw pair metadata when present."""
    a_is_left = pair.get("image_a_is_left")
    b_is_left = pair.get("image_b_is_left")
    if isinstance(a_is_left, bool) and isinstance(b_is_left, bool):
        return ("L" if a_is_left else "R") + ("L" if b_is_left else "R")

    image_a = pair.get("image_a", pair.get("left_image", ""))
    image_b = pair.get("image_b", pair.get("right_image", ""))
    a_side = "L" if "image_00" in image_a or image_a.startswith("left") else "R"
    b_side = "L" if "image_00" in image_b or image_b.startswith("left") else "R"
    return a_side + b_side


def add_point(bucket: dict[str, list[list[float]]], is_left: bool, point: list[float], ptype: str) -> None:
    side = "left" if is_left else "right"
    if len(point) >= 2 and math.isfinite(point[0]) and math.isfinite(point[1]):
        bucket[side].append([float(point[0]), float(point[1])])
        bucket[f"{side}_{ptype}"].append([float(point[0]), float(point[1])])


def collect_points(data: dict, max_score: float | None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    pair_counts: dict[str, int] = defaultdict(int)
    match_counts: dict[str, int] = defaultdict(int)

    for pair in data.get("pairs", []):
        ptype = pair_type(pair)
        pair_counts[ptype] += 1
        a_is_left = pair.get("image_a_is_left", ptype[0] == "L")
        b_is_left = pair.get("image_b_is_left", ptype[1] == "L")

        for match in pair.get("matches", []):
            score = float(match.get("score", 0.0))
            if max_score is not None and score > max_score:
                continue
            if "left" not in match or "right" not in match:
                continue

            match_counts[ptype] += 1
            add_point(buckets, bool(a_is_left), match["left"], ptype)
            add_point(buckets, bool(b_is_left), match["right"], ptype)

    arrays = {}
    for key, points in buckets.items():
        arrays[key] = np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)

    counts = {f"pairs_{k}": int(v) for k, v in pair_counts.items()}
    counts.update({f"matches_{k}": int(v) for k, v in match_counts.items()})
    return arrays, counts


def infer_image_size(points_by_side: dict[str, np.ndarray], width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None and height is not None:
        return width, height

    all_points = []
    for side in ("left", "right"):
        pts = points_by_side.get(side)
        if pts is not None and len(pts):
            all_points.append(pts)
    if not all_points:
        return width or 1, height or 1

    pts = np.vstack(all_points)
    inferred_w = int(math.ceil(float(np.max(pts[:, 0])) + 1))
    inferred_h = int(math.ceil(float(np.max(pts[:, 1])) + 1))
    return width or inferred_w, height or inferred_h


def grid_stats(points: np.ndarray, width: int, height: int, cols: int, rows: int) -> tuple[np.ndarray, dict]:
    if len(points) == 0:
        heat = np.zeros((rows, cols), dtype=int)
    else:
        x_edges = np.linspace(0, width, cols + 1)
        y_edges = np.linspace(0, height, rows + 1)
        heat, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[y_edges, x_edges])
        heat = heat.astype(int)

    occupied = heat > 0
    nonzero = heat[occupied]
    total = int(heat.sum())
    stats = {
        "observations": total,
        "occupied_cells": int(occupied.sum()),
        "total_cells": int(rows * cols),
        "coverage": float(occupied.sum() / max(rows * cols, 1)),
        "empty_cells": int((~occupied).sum()),
        "mean_per_occupied_cell": float(nonzero.mean()) if nonzero.size else 0.0,
        "median_per_occupied_cell": float(np.median(nonzero)) if nonzero.size else 0.0,
        "max_cell_count": int(nonzero.max()) if nonzero.size else 0,
        "top_half_ratio": float(heat[: rows // 2, :].sum() / total) if total else 0.0,
        "bottom_half_ratio": float(heat[rows // 2 :, :].sum() / total) if total else 0.0,
        "upper_third_ratio": float(heat[: max(rows // 3, 1), :].sum() / total) if total else 0.0,
        "lower_third_ratio": float(heat[rows - max(rows // 3, 1) :, :].sum() / total) if total else 0.0,
    }
    return heat, stats


def plot_side_scatter(ax, arrays: dict[str, np.ndarray], side: str, width: int, height: int) -> None:
    colors = {
        "LR": "#1f77b4",
        "LL": "#2ca02c",
        "RR": "#d62728",
        "RL": "#9467bd",
    }
    for ptype, color in colors.items():
        pts = arrays.get(f"{side}_{ptype}", np.empty((0, 2)))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.18, c=color, label=ptype, rasterized=True)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"{side.capitalize()} camera observations")
    ax.legend(markerscale=4, fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, color="#d9d9d9", linewidth=0.4, alpha=0.6)


def plot_heatmap(ax, heat: np.ndarray, side: str, stats: dict) -> None:
    im = ax.imshow(heat, cmap="magma", origin="upper", aspect="auto")
    ax.set_title(
        f"{side.capitalize()} grid count "
        f"(coverage {stats['coverage'] * 100:.1f}%, max {stats['max_cell_count']})"
    )
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")

    rows, cols = heat.shape
    if rows <= 8 and cols <= 16:
        threshold = heat.max() * 0.55 if heat.size and heat.max() else 1
        for r in range(rows):
            for c in range(cols):
                value = int(heat[r, c])
                if value == 0:
                    continue
                ax.text(
                    c,
                    r,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="observation count")


def default_output_path(input_path: Path) -> Path:
    if input_path.parent.name == "match_points":
        return input_path.parent.parent / f"{input_path.stem}_point_distribution.png"
    return input_path.with_name(f"{input_path.stem}_point_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot match point spatial distribution.")
    parser.add_argument("--input", required=True, help="Input matches JSON.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--stats_output", default=None, help="Output JSON stats path.")
    parser.add_argument("--max_score", type=float, default=None, help="Keep only matches with score <= this value.")
    parser.add_argument("--width", type=int, default=None, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Image height in pixels.")
    parser.add_argument("--grid_cols", type=int, default=12, help="Number of heatmap columns.")
    parser.add_argument("--grid_rows", type=int, default=6, help="Number of heatmap rows.")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    arrays, counts = collect_points(data, args.max_score)
    width, height = infer_image_size(arrays, args.width, args.height)

    left_heat, left_stats = grid_stats(arrays.get("left", np.empty((0, 2))), width, height, args.grid_cols, args.grid_rows)
    right_heat, right_stats = grid_stats(arrays.get("right", np.empty((0, 2))), width, height, args.grid_cols, args.grid_rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    score_text = "all scores" if args.max_score is None else f"score <= {args.max_score:g}"
    fig.suptitle(
        f"Match point distribution: {input_path.name}\n"
        f"{score_text}, image {width}x{height}, grid {args.grid_cols}x{args.grid_rows}",
        fontsize=12,
    )

    plot_side_scatter(axes[0, 0], arrays, "left", width, height)
    plot_heatmap(axes[0, 1], left_heat, "left", left_stats)
    plot_side_scatter(axes[1, 0], arrays, "right", width, height)
    plot_heatmap(axes[1, 1], right_heat, "right", right_stats)

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "max_score": args.max_score,
        "image_size": {"width": width, "height": height},
        "grid": {"cols": args.grid_cols, "rows": args.grid_rows},
        "pair_and_match_counts": counts,
        "left": left_stats,
        "right": right_stats,
    }

    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved plot : {output_path}")
    print(f"Saved stats: {stats_path}")
    print(
        "Left  coverage={:.1f}% observations={} upper_third={:.1f}% lower_third={:.1f}%".format(
            left_stats["coverage"] * 100,
            left_stats["observations"],
            left_stats["upper_third_ratio"] * 100,
            left_stats["lower_third_ratio"] * 100,
        )
    )
    print(
        "Right coverage={:.1f}% observations={} upper_third={:.1f}% lower_third={:.1f}%".format(
            right_stats["coverage"] * 100,
            right_stats["observations"],
            right_stats["upper_third_ratio"] * 100,
            right_stats["lower_third_ratio"] * 100,
        )
    )


if __name__ == "__main__":
    main()
