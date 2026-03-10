#!/usr/bin/env python3
"""
plot_ba_history.py — 绘制 BA 优化历史的所有参数误差曲线

从 offline_ba_result.json 的 optimization_history 字段读取每次优化迭代的误差，
绘制 8 张子图（重投影误差、旋转误差、左右相机焦距误差、基线误差、平移误差），
并在全局优化（Global BA）发生处标注红色星形和竖线。

用法：
    python plot_ba_history.py [--input RESULT_JSON] [--output PLOT_FILE]

示例：
    python stereo_calib/scripts/plot_ba_history.py \
        --input stereo_calib/result/offline_ba_result.json \
        --output stereo_calib/result/ba_history.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── 配色方案 ──────────────────────────────────────────────────────────────────
COLOR_LINE   = "#4c72b0"   # 主折线（增量 BA）
COLOR_INCR   = "#4c72b0"   # 增量 BA 点
COLOR_GLOBAL = "#d62728"   # 全局 BA 点 / 标注
COLOR_VLINE  = "#d62728"   # 全局 BA 竖线
BG_AXES      = "#ffffff"
BG_FIG       = "#f4f6f8"


def is_global(stage: str) -> bool:
    return "global ba" in stage.lower()


def load_data(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"错误：文件不存在 —— {json_path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("optimization_history")
    if not history:
        print("错误：JSON 中未找到 optimization_history 字段。", file=sys.stderr)
        sys.exit(1)
    return history, data


def extract_series(history):
    """从 optimization_history 中提取所有需要绘制的时间序列。"""
    stages        = []
    reproj        = []
    rot_err_deg   = []
    left_fx_err   = []
    left_fy_err   = []
    right_fx_err  = []
    right_fy_err  = []
    baseline_err  = []
    trans_err_x   = []
    trans_err_y   = []
    trans_err_z   = []

    for item in history:
        stages.append(item["stage"])
        reproj.append(item.get("reproj_error", float("nan")))

        dvg = item.get("diff_vs_gt", {})
        ext = dvg.get("extrinsics", {})
        lft = dvg.get("left", {})
        rgt = dvg.get("right", {})

        rot_err_deg.append(ext.get("rotation_error_deg", float("nan")))
        baseline_err.append(abs(ext.get("baseline", float("nan"))))

        t = ext.get("t", [float("nan")] * 3)
        trans_err_x.append(abs(t[0]))
        trans_err_y.append(abs(t[1]))
        trans_err_z.append(abs(t[2]))

        left_fx_err.append(abs(lft.get("fx", float("nan"))))
        left_fy_err.append(abs(lft.get("fy", float("nan"))))
        right_fx_err.append(abs(rgt.get("fx", float("nan"))))
        right_fy_err.append(abs(rgt.get("fy", float("nan"))))

    idx = list(range(1, len(stages) + 1))
    global_mask = [is_global(s) for s in stages]

    return dict(
        stages=stages, idx=idx, global_mask=global_mask,
        reproj=reproj, rot_err_deg=rot_err_deg,
        left_fx_err=left_fx_err, left_fy_err=left_fy_err,
        right_fx_err=right_fx_err, right_fy_err=right_fy_err,
        baseline_err=baseline_err,
        trans_err_x=trans_err_x, trans_err_y=trans_err_y, trans_err_z=trans_err_z,
    )


def short_label(stage: str) -> str:
    if "Final Global BA" in stage:
        return "Final\nGlobal"
    if "Periodic Global BA" in stage:
        frame = stage.split("Frame")[-1].strip()
        return f"GlobalBA\nF{frame}"
    frame = stage.split("Frame")[-1].strip()
    return f"F{frame}"


def _draw_panel(ax, idx, values, global_mask, ylabel, unit="",
                color=COLOR_LINE, extra_lines=None):
    """
    在单个子图中绘制一条（或多条）折线，并标注全局 BA。

    Parameters
    ----------
    extra_lines : list of (values, label, color) | None
        若需要在同一子图中绘制额外折线（如 tx/ty/tz），传入此参数。
    """
    ax.set_facecolor(BG_AXES)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    global_x = [i for i, g in zip(idx, global_mask) if g]
    global_y = [v for v, g in zip(values, global_mask) if g]
    incr_x   = [i for i, g in zip(idx, global_mask) if not g]
    incr_y   = [v for v, g in zip(values, global_mask) if not g]

    # 全局 BA 竖线（最底层）
    for xi in global_x:
        ax.axvline(xi, color=COLOR_VLINE, linewidth=0.8,
                   linestyle="--", alpha=0.35, zorder=1)

    # 主折线
    ax.plot(idx, values, color=color, linewidth=1.5, zorder=2)
    ax.scatter(incr_x, incr_y, color=color, s=35, zorder=3, label="Incremental BA")
    ax.scatter(global_x, global_y, color=COLOR_GLOBAL, s=100,
               marker="*", zorder=4, label="Global BA")

    # 额外折线（如 tx/ty/tz）
    if extra_lines:
        for vals, label, clr in extra_lines:
            ax.plot(idx, vals, linewidth=1.2, color=clr, linestyle="--",
                    zorder=2, label=label)

    full_ylabel = f"{ylabel}" + (f" ({unit})" if unit else "")
    ax.set_ylabel(full_ylabel, fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_all(series, data, output_path):
    stages      = series["stages"]
    idx         = series["idx"]
    global_mask = series["global_mask"]
    n           = len(stages)

    xlabels = [short_label(s) for s in stages]

    # ── 画布：4 行 × 2 列 ─────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
    fig.patch.set_facecolor(BG_FIG)

    num_frames = data.get("num_frames", "?")
    num_tracks = data.get("num_tracks", "?")
    fig.suptitle(
        f"BA Optimization History — All Parameter Errors\n"
        f"({num_frames} frames, {num_tracks} tracks, {n} stages)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # ── 子图定义：(ax, values, ylabel, unit, extra_lines) ────────────
    panels = [
        (axes[0, 0], series["reproj"],       "Reprojection Error",  "px",  None),
        (axes[0, 1], series["rot_err_deg"],   "Rotation Error",      "deg", None),
        (axes[1, 0], series["left_fx_err"],   "Left  |Δfx|",         "px",  None),
        (axes[1, 1], series["left_fy_err"],   "Left  |Δfy|",         "px",  None),
        (axes[2, 0], series["right_fx_err"],  "Right |Δfx|",         "px",  None),
        (axes[2, 1], series["right_fy_err"],  "Right |Δfy|",         "px",  None),
        (axes[3, 0], series["baseline_err"],  "Baseline |Δt_x|",     "m",   None),
        (axes[3, 1], series["trans_err_x"],   "Translation Error",   "m",
         [
             (series["trans_err_y"], "|Δty|", "#2ca02c"),
             (series["trans_err_z"], "|Δtz|", "#ff7f0e"),
         ]),
    ]

    for ax, vals, ylabel, unit, extra in panels:
        _draw_panel(ax, idx, vals, global_mask, ylabel, unit, extra_lines=extra)

    # ── 共享 x 轴刻度 ─────────────────────────────────────────────────
    for row in axes:
        for ax in row:
            ax.set_xticks(idx)
            ax.set_xticklabels(xlabels, fontsize=6.5, rotation=45, ha="right")
            ax.set_xlim(0.5, n + 0.5)

    # 最后一行才显示 x 轴标签说明
    for ax in axes[-1]:
        ax.set_xlabel("Optimization Stage", fontsize=8.5)

    # ── 图例（只在右上子图显示一次）─────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=COLOR_INCR,   marker="o", markersize=5,
                   linewidth=1.5, label="Incremental BA"),
        plt.Line2D([0], [0], color=COLOR_GLOBAL, marker="*", markersize=9,
                   linewidth=0,   label="Global BA"),
    ]
    axes[3, 1].legend(
        handles=handles + [
            plt.Line2D([0], [0], color=COLOR_LINE,    linewidth=1.5, label="|Δtx|"),
            plt.Line2D([0], [0], color="#2ca02c", linewidth=1.2,
                       linestyle="--", label="|Δty|"),
            plt.Line2D([0], [0], color="#ff7f0e", linewidth=1.2,
                       linestyle="--", label="|Δtz|"),
        ],
        fontsize=7.5, loc="upper right", framealpha=0.85, edgecolor="#cccccc",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"图表已保存至：{out}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="绘制 BA 优化历史的所有参数误差曲线"
    )
    parser.add_argument(
        "--input", "-i",
        default="stereo_calib/result/offline_ba_result.json",
        help="result JSON 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="stereo_calib/result/ba_history.png",
        help="输出图片路径（默认：stereo_calib/result/ba_history.png）",
    )
    args = parser.parse_args()

    history, data = load_data(args.input)
    series = extract_series(history)
    plot_all(series, data, args.output)


if __name__ == "__main__":
    main()
