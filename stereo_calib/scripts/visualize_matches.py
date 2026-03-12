"""
visualize_matches.py -- Visualize SuperPoint match information from matches.json

Usage:
    # Overview: connectivity heatmap + score distribution + per-frame stats
    python visualize_matches.py --input stereo_calib/data/matches.json

    # Change valid-match threshold
    python visualize_matches.py --input matches.json --max_score 0.3

    # Show detailed view for a specific pair
    python visualize_matches.py --input matches.json --pair left_000-left_015

    # Save to file instead of showing window
    python visualize_matches.py --input matches.json --output overview.png
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_frame_id(name):
    """'left_015.png' -> '015'"""
    return name.split("_")[1].split(".")[0]


def get_side(name):
    """'left_015.png' -> 'L',  'right_015.png' -> 'R'"""
    return "L" if name.startswith("left") else "R"


def classify_pair(p):
    """Returns 'stereo' / 'LL' / 'LR' / 'RR'"""
    fa, fb = get_frame_id(p["left_image"]), get_frame_id(p["right_image"])
    sa, sb = get_side(p["left_image"]), get_side(p["right_image"])
    if fa == fb:
        return "stereo"
    return sa + sb


def load_matches(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Figure 1: Overview (4 panels)
# ─────────────────────────────────────────────

def plot_overview(data, max_score, ax_heat, ax_score, ax_frame, ax_scatter):
    pairs = data["pairs"]

    frame_ids = sorted(
        {get_frame_id(p["left_image"]) for p in pairs}
        | {get_frame_id(p["right_image"]) for p in pairs}
    )
    frame_idx = {f: i for i, f in enumerate(frame_ids)}
    n = len(frame_ids)

    heat = np.zeros((n, n), dtype=float)
    all_scores = []
    frame_match_count = defaultdict(int)
    pair_stats = []

    for p in pairs:
        fa, fb = get_frame_id(p["left_image"]), get_frame_id(p["right_image"])
        ptype = classify_pair(p)

        valid = [m for m in p["matches"] if m["score"] <= max_score]
        n_valid = len(valid)
        scores = [m["score"] for m in p["matches"]]
        all_scores.extend(scores)

        if ptype == "LL" and n_valid > 0:
            i, j = frame_idx[fa], frame_idx[fb]
            heat[i, j] = n_valid
            heat[j, i] = n_valid

        if n_valid > 0:
            frame_match_count[fa] += n_valid
            frame_match_count[fb] += n_valid

        avg_score = float(np.mean(scores)) if scores else 0.0
        pair_stats.append((ptype, len(p["matches"]), n_valid, avg_score))

    # ── Panel A: L-L inter-frame connectivity heatmap ──
    im = ax_heat.imshow(heat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax_heat,
                 label=f"valid L-L matches (score <= {max_score})")
    tick_labels = [f.lstrip("0") or "0" for f in frame_ids]
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(tick_labels, fontsize=6)
    ax_heat.set_title("L-L Inter-frame Connectivity (rotation deg)", fontsize=10)
    ax_heat.set_xlabel("Frame")
    ax_heat.set_ylabel("Frame")

    # Annotate cells with count where count >= registration threshold (8)
    for i in range(n):
        for j in range(n):
            if heat[i, j] >= 8:
                ax_heat.text(j, i, int(heat[i, j]),
                             ha="center", va="center", fontsize=4,
                             color="black" if heat[i, j] < heat.max() * 0.6 else "white")

    # ── Panel B: Match score histogram (all pairs) ──
    bins = np.linspace(0, max(all_scores) * 1.05, 60)
    ax_score.hist(all_scores, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
    ax_score.axvline(max_score, color="crimson", linestyle="--",
                     linewidth=1.5, label=f"max_score={max_score}")
    ax_score.set_title("Match Score Distribution (all pairs)", fontsize=10)
    ax_score.set_xlabel("Score (descriptor distance, lower = better)")
    ax_score.set_ylabel("Match count")
    ax_score.legend(fontsize=8)

    # ── Panel C: Valid match count per frame ──
    counts = [frame_match_count.get(f, 0) for f in frame_ids]
    colors = ["tomato" if c == 0 else "steelblue" for c in counts]
    ax_frame.bar(range(n), counts, color=colors, width=0.7)
    ax_frame.set_xticks(range(n))
    ax_frame.set_xticklabels(tick_labels, fontsize=7, rotation=90)
    ax_frame.set_title(f"Valid Matches per Frame (score <= {max_score})", fontsize=10)
    ax_frame.set_xlabel("Frame (rotation deg)")
    ax_frame.set_ylabel("Valid match count")
    ax_frame.set_ylim(bottom=0)

    isolated = [tick_labels[i] for i, c in enumerate(counts) if c == 0]
    if isolated:
        ax_frame.text(0.99, 0.97, f"Isolated frames (0 matches): {', '.join(isolated)}",
                      transform=ax_frame.transAxes, ha="right", va="top",
                      fontsize=7, color="tomato")

    # ── Panel D: Raw match count vs valid match count scatter ──
    type_colors = {"stereo": "green", "LL": "steelblue", "LR": "orange", "RR": "purple"}
    for ptype, raw, valid, _ in pair_stats:
        c = type_colors.get(ptype, "gray")
        ax_scatter.scatter(raw, valid, c=c, s=6, alpha=0.4)

    for ptype, c in type_colors.items():
        ax_scatter.scatter([], [], c=c, s=20, label=ptype)
    max_raw = max(r for _, r, _, _ in pair_stats)
    ax_scatter.plot([0, max_raw], [0, max_raw], "k--", linewidth=0.8, label="y=x")
    ax_scatter.axhline(y=15, color="crimson", linestyle=":", linewidth=1,
                       label="min_pair_inliers=15")
    ax_scatter.set_title("Raw Matches vs Valid Matches per Pair", fontsize=10)
    ax_scatter.set_xlabel("Raw match count (num_matches)")
    ax_scatter.set_ylabel(f"Valid match count (score <= {max_score})")
    ax_scatter.legend(fontsize=7, markerscale=2)


# ─────────────────────────────────────────────
# Figure 2: Single-pair detail
# ─────────────────────────────────────────────

def plot_pair_detail(data, pair_key, max_score):
    """pair_key: 'left_000-left_015' format (without .png)"""
    parts = pair_key.split("-", 1)
    if len(parts) != 2:
        print(f"[ERROR] --pair format should be 'left_000-left_015', got: {pair_key}")
        return None

    target_left = parts[0] + ".png"
    target_right = parts[1] + ".png"
    pair = None
    for p in data["pairs"]:
        if p["left_image"] == target_left and p["right_image"] == target_right:
            pair = p
            break

    if pair is None:
        print(f"[ERROR] Pair not found: {target_left} <-> {target_right}")
        print("  Available examples:")
        for p in data["pairs"][:5]:
            print(f"    {p['left_image'].replace('.png','')} - {p['right_image'].replace('.png','')}")
        return None

    matches = pair["matches"]
    scores = np.array([m["score"] for m in matches])
    lx = np.array([m["left"][0] for m in matches])
    ly = np.array([m["left"][1] for m in matches])
    rx = np.array([m["right"][0] for m in matches])
    ry = np.array([m["right"][1] for m in matches])
    valid = scores <= max_score

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Pair Detail:  {target_left}  <->  {target_right}\n"
        f"Total matches: {len(matches)}   |   Valid (score<={max_score}): {valid.sum()}",
        fontsize=12
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── A: Match point distribution in left image space ──
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.scatter(lx[~valid], ly[~valid], s=4, c="lightgray", alpha=0.5, label="invalid")
    sc = ax_l.scatter(lx[valid], ly[valid], s=8, c=scores[valid],
                      cmap="plasma_r", vmin=0, vmax=max_score, alpha=0.8, label="valid")
    plt.colorbar(sc, ax=ax_l, label="score")
    ax_l.set_title(f"Match pts in left image\n({target_left})")
    ax_l.set_xlabel("x (px)")
    ax_l.set_ylabel("y (px)")
    ax_l.invert_yaxis()
    ax_l.set_aspect("equal")
    ax_l.legend(fontsize=7, markerscale=2)

    # ── B: Match point distribution in right image space ──
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.scatter(rx[~valid], ry[~valid], s=4, c="lightgray", alpha=0.5)
    ax_r.scatter(rx[valid], ry[valid], s=8, c=scores[valid],
                 cmap="plasma_r", vmin=0, vmax=max_score, alpha=0.8)
    ax_r.set_title(f"Match pts in right image\n({target_right})")
    ax_r.set_xlabel("x (px)")
    ax_r.set_ylabel("y (px)")
    ax_r.invert_yaxis()
    ax_r.set_aspect("equal")

    # ── C: Displacement (dx, dy) -- epipolar constraint check ──
    ax_dx = fig.add_subplot(gs[0, 2])
    dx = rx - lx
    dy = ry - ly
    ax_dx.scatter(dx[~valid], dy[~valid], s=5, c="lightgray", alpha=0.4)
    sc2 = ax_dx.scatter(dx[valid], dy[valid], s=8, c=scores[valid],
                        cmap="plasma_r", vmin=0, vmax=max_score, alpha=0.8)
    ax_dx.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_dx.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_dx.set_title("Match displacement (dx, dy)\n[pure rotation -> dy~=0]")
    ax_dx.set_xlabel("dx = right_x - left_x (px)")
    ax_dx.set_ylabel("dy = right_y - left_y (px)")
    plt.colorbar(sc2, ax=ax_dx, label="score")

    # ── D: Score histogram ──
    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, scores.max() * 1.05, 40)
    ax_hist.hist(scores, bins=bins, color="steelblue", edgecolor="none")
    ax_hist.axvline(max_score, color="crimson", linestyle="--",
                    linewidth=1.5, label=f"max_score={max_score}")
    ax_hist.set_title("Score Histogram")
    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)

    # ── E: Horizontal disparity (dx) distribution ──
    ax_dispx = fig.add_subplot(gs[1, 1])
    if valid.sum() > 0:
        ax_dispx.hist(dx[valid], bins=40, color="seagreen", edgecolor="none", alpha=0.8)
        ax_dispx.axvline(float(np.median(dx[valid])), color="crimson", linestyle="--",
                         linewidth=1.5,
                         label=f"median={float(np.median(dx[valid])):.1f}px")
        ax_dispx.legend(fontsize=8)
    ax_dispx.set_title("Valid Match dx Distribution")
    ax_dispx.set_xlabel("dx (px)")
    ax_dispx.set_ylabel("Count")

    # ── F: Match connection lines (sampled) ──
    ax_conn = fig.add_subplot(gs[1, 2])
    valid_idx = np.where(valid)[0]
    sample = (valid_idx if len(valid_idx) <= 80
              else np.random.choice(valid_idx, 80, replace=False))
    cmap = plt.get_cmap("plasma_r")
    for idx in sample:
        color = cmap(scores[idx] / max(max_score, 1e-6))
        ax_conn.plot([0, 1], [ly[idx], ry[idx]],
                     color=color, alpha=0.4, linewidth=0.7)
    ax_conn.scatter([0] * len(sample), ly[sample], s=4, c="steelblue", zorder=3)
    ax_conn.scatter([1] * len(sample), ry[sample], s=4, c="orange", zorder=3)
    ax_conn.set_xticks([0, 1])
    ax_conn.set_xticklabels(["left", "right"])
    ax_conn.set_ylabel("y (px)")
    ax_conn.invert_yaxis()
    ax_conn.set_title(f"Match Connection Lines (sample {len(sample)})")

    return fig


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize matches.json match information")
    parser.add_argument("--input", required=True, help="Path to matches.json")
    parser.add_argument("--max_score", type=float, default=0.4,
                        help="Valid-match score threshold (default 0.4)")
    parser.add_argument("--pair", default=None,
                        help="Show detail for one pair, e.g. left_000-left_015")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path (default: auto-save to result/ folder)")
    args = parser.parse_args()

    data = load_matches(args.input)
    print(f"Loaded {len(data['pairs'])} pairs")

    if args.pair:
        fig = plot_pair_detail(data, args.pair, args.max_score)
        if fig is None:
            sys.exit(1)
    else:
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle("SuperPoint Match Overview", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
        ax_heat    = fig.add_subplot(gs[0, 0])
        ax_score   = fig.add_subplot(gs[0, 1])
        ax_frame   = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[1, 1])
        plot_overview(data, args.max_score, ax_heat, ax_score, ax_frame, ax_scatter)

    # Determine output path: --output > auto result/ folder > show window
    out_path = args.output
    if out_path is None:
        result_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                  "..", "result")
        os.makedirs(result_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        if args.pair:
            safe_key = args.pair.replace("/", "_")
            filename = f"{stem}_pair_{safe_key}.png"
        else:
            filename = f"{stem}_overview.png"
        out_path = os.path.join(result_dir, filename)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
