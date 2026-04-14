#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


BG_FIG = "#f4f6f8"
BG_AXES = "#ffffff"
HEADER_COLOR = "#d9e6f2"
BEST_LOW_COLOR = "#e8f5e9"
BEST_HIGH_COLOR = "#fff3e0"
DEFAULT_FONT_SIZE = 10


def load_json(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        print(f"错误：文件不存在 —— {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_summary(data: dict, path_label: str) -> dict:
    summary = data.get("summary")
    if not isinstance(summary, dict):
        print(f"错误：{path_label} 缺少 summary 字段。", file=sys.stderr)
        sys.exit(1)
    return summary


def fmt_bool(value) -> str:
    return "Yes" if bool(value) else "No"


def fmt_int(value) -> str:
    return str(int(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else "N/A"


def fmt_float(value, digits: int = 6) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def collect_row(label: str, data: dict) -> dict:
    summary = require_summary(data, label)
    return {
        "Threshold": label,
        "Success": fmt_bool(data.get("success", True)),
        "Tracks": data.get("num_tracks"),
        "Observations": data.get("num_observations"),
        "Final Reproj": data.get("final_reproj_error"),
        "Mean Reproj": summary.get("avg_reproj_error_px"),
        "Mean Rotation": summary.get("avg_rotation_error_deg"),
        "Mean Focal": summary.get("avg_focal_error_px"),
        "Mean Baseline": summary.get("avg_baseline_error_m"),
    }


def build_rows(inputs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for label, path in inputs:
        data = load_json(path)
        rows.append(collect_row(label, data))
    return rows


def colorize_best_cells(table, rows: list[dict], columns: list[str]) -> None:
    lower_better = {"Final Reproj", "Mean Reproj", "Mean Rotation", "Mean Focal", "Mean Baseline"}
    higher_better = {"Tracks", "Observations"}

    for col_idx, col_name in enumerate(columns):
        if col_name not in lower_better and col_name not in higher_better:
            continue

        candidates = []
        for row_idx, row in enumerate(rows, start=1):
            value = row[col_name]
            if isinstance(value, (int, float)) and math.isfinite(value):
                candidates.append((row_idx, float(value)))

        if not candidates:
            continue

        target = min(v for _, v in candidates) if col_name in lower_better else max(v for _, v in candidates)
        color = BEST_LOW_COLOR if col_name in lower_better else BEST_HIGH_COLOR
        for row_idx, value in candidates:
            if value == target:
                table[(row_idx, col_idx)].set_facecolor(color)


def render_table(scene: str, rows: list[dict], output: str) -> None:
    columns = [
        "Threshold",
        "Success",
        "Tracks",
        "Observations",
        "Final Reproj",
        "Mean Reproj",
        "Mean Rotation",
        "Mean Focal",
        "Mean Baseline",
    ]

    table_text = []
    for row in rows:
        table_text.append([
            row["Threshold"],
            row["Success"],
            fmt_int(row["Tracks"]),
            fmt_int(row["Observations"]),
            fmt_float(row["Final Reproj"]),
            fmt_float(row["Mean Reproj"]),
            fmt_float(row["Mean Rotation"]),
            fmt_float(row["Mean Focal"]),
            fmt_float(row["Mean Baseline"]),
        ])

    fig, ax = plt.subplots(figsize=(15, 3.2))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AXES)
    ax.axis("off")

    fig.suptitle(f"BA Threshold Comparison — {scene}", fontsize=14, fontweight="bold", y=0.96)

    table = ax.table(
        cellText=table_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(DEFAULT_FONT_SIZE)
    table.scale(1.05, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(weight="bold")

    colorize_best_cells(table, rows, columns)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比表已保存至：{out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 3px/2px/1px BA 结果对比表")
    parser.add_argument("--scene", required=True, help="场景名称")
    parser.add_argument("--input-3px", required=True, help="3px BA 结果 JSON")
    parser.add_argument("--input-2px", required=True, help="2px BA 结果 JSON")
    parser.add_argument("--input-1px", required=True, help="1px BA 结果 JSON")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    args = parser.parse_args()

    rows = build_rows([
        ("3px", args.input_3px),
        ("2px", args.input_2px),
        ("1px", args.input_1px),
    ])
    render_table(args.scene, rows, args.output)


if __name__ == "__main__":
    main()
