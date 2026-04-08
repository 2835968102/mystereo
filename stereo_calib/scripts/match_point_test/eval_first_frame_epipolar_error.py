import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PARAM_FILE = Path("/home/hello/pml/mycalib/blender-file/stereo_panoramic_01/camera_params_0000.json")
DEFAULT_MATCHES_FILE = Path("/home/hello/pml/mycalib/stereo_calib/result/panoramic_01_matches.json")
DEFAULT_OUTPUT_JSON = Path("/home/hello/pml/mycalib/match_point_test/panoramic_01_frame0000_epipolar_error.json")
DEFAULT_OUTPUT_PLOT = Path("/home/hello/pml/mycalib/match_point_test/panoramic_01_frame0000_epipolar_error.png")
EPS = 1e-12


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate first-frame stereo epipolar errors")
    parser.add_argument("--param_file", default=str(DEFAULT_PARAM_FILE), help="Path to first-frame camera parameter JSON")
    parser.add_argument("--matches_file", default=str(DEFAULT_MATCHES_FILE), help="Path to matches JSON")
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON), help="Path to output JSON summary")
    parser.add_argument("--plot_file", default=str(DEFAULT_OUTPUT_PLOT), help="Path to output visualization image")
    parser.add_argument("--error_threshold", type=float, default=2.0, help="Highlight matches with d_sym >= threshold (pixels)")
    return parser.parse_args()


def build_K(intr: dict) -> np.ndarray:
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = t.tolist()
    return np.array(
        [
            [0.0, -tz, ty],
            [tz, 0.0, -tx],
            [-ty, tx, 0.0],
        ],
        dtype=np.float64,
    )


def compute_E_F(K1: np.ndarray, K2: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = skew(t) @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return E, F


def find_target_pair(matches_data: dict, left_name: str, right_name: str):
    candidates = []
    for pair in matches_data["pairs"]:
        a = pair["left_image"]
        b = pair["right_image"]
        if a == left_name and b == right_name:
            candidates.append((pair, False))
        elif a == right_name and b == left_name:
            candidates.append((pair, True))

    if len(candidates) != 1:
        names = [(p["left_image"], p["right_image"]) for p in matches_data["pairs"]]
        raise ValueError(
            f"Expected exactly one pair for {left_name} <-> {right_name}, got {len(candidates)}. "
            f"Available pair examples: {names[:10]}"
        )

    return candidates[0]


def safe_point_line_distance(line: np.ndarray, point_h: np.ndarray) -> float:
    denom = math.hypot(float(line[0]), float(line[1]))
    if denom < EPS:
        return float("nan")
    return abs(float(point_h @ line)) / denom


def compute_point_metrics(pair: dict, reverse_pair: bool, F: np.ndarray):
    algebraic_errors = []
    d1_list = []
    d2_list = []
    dsym_list = []
    points = []
    invalid_line_count = 0

    for idx, match in enumerate(pair["matches"]):
        if reverse_pair:
            x1_xy = np.array(match["right"], dtype=np.float64)
            x2_xy = np.array(match["left"], dtype=np.float64)
        else:
            x1_xy = np.array(match["left"], dtype=np.float64)
            x2_xy = np.array(match["right"], dtype=np.float64)

        x1 = np.array([x1_xy[0], x1_xy[1], 1.0], dtype=np.float64)
        x2 = np.array([x2_xy[0], x2_xy[1], 1.0], dtype=np.float64)

        l2 = F @ x1
        l1 = F.T @ x2
        algebraic_error = float(x2 @ l2)
        d2 = safe_point_line_distance(l2, x2)
        d1 = safe_point_line_distance(l1, x1)
        d_sym = d1 + d2 if not (math.isnan(d1) or math.isnan(d2)) else float("nan")

        if math.isnan(d1) or math.isnan(d2):
            invalid_line_count += 1

        algebraic_errors.append(algebraic_error)
        d1_list.append(d1)
        d2_list.append(d2)
        dsym_list.append(d_sym)
        points.append(
            {
                "index": idx,
                "score": float(match["score"]),
                "x1": x1_xy.tolist(),
                "x2": x2_xy.tolist(),
                "algebraic_error": algebraic_error,
                "line1": l1.tolist(),
                "line2": l2.tolist(),
                "d1": d1,
                "d2": d2,
                "d_sym": d_sym,
            }
        )

    return {
        "points": points,
        "algebraic_errors": np.array(algebraic_errors, dtype=np.float64),
        "d1": np.array(d1_list, dtype=np.float64),
        "d2": np.array(d2_list, dtype=np.float64),
        "d_sym": np.array(dsym_list, dtype=np.float64),
        "invalid_line_count": invalid_line_count,
    }


def summarize(values: np.ndarray, absolute: bool = False) -> dict:
    arr = np.abs(values) if absolute else values
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "rms": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "rms": float(np.sqrt(np.mean(valid ** 2))),
        "max": float(np.max(valid)),
    }


def matrix_rank_singular_values(F: np.ndarray) -> list[float]:
    return [float(v) for v in np.linalg.svd(F, compute_uv=False)]


def load_image(path: Path) -> np.ndarray:
    return plt.imread(path)


def draw_matches(ax, image: np.ndarray, points: list[dict], title: str, key: str, threshold: float):
    ax.imshow(image)
    normal_x = []
    normal_y = []
    bad_x = []
    bad_y = []

    for point in points:
        x, y = point[key]
        if point["d_sym"] >= threshold:
            bad_x.append(x)
            bad_y.append(y)
        else:
            normal_x.append(x)
            normal_y.append(y)

    if normal_x:
        ax.scatter(normal_x, normal_y, s=10, c="lime", alpha=0.6, label=f"d_sym < {threshold:g}px")
    if bad_x:
        ax.scatter(bad_x, bad_y, s=22, facecolors="none", edgecolors="red", linewidths=1.0, label=f"d_sym >= {threshold:g}px")

    ax.set_title(title)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.legend(loc="upper right", fontsize=8)



def plot_large_error_matches(scene_dir: Path, left_name: str, right_name: str, points: list[dict], threshold: float, output_path: Path):
    left_img = load_image(scene_dir / left_name)
    right_img = load_image(scene_dir / right_name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    draw_matches(axes[0], left_img, points, f"Left image: {left_name}", "x1", threshold)
    draw_matches(axes[1], right_img, points, f"Right image: {right_name}", "x2", threshold)

    large_error_count = sum(1 for point in points if point["d_sym"] >= threshold)
    fig.suptitle(
        f"First-frame stereo matches with large epipolar error\n"
        f"Highlighted: d_sym >= {threshold:g} px | total={len(points)} | bad={large_error_count}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def main():
    args = parse_args()
    param_path = Path(args.param_file)
    matches_path = Path(args.matches_file)
    output_path = Path(args.output_json)
    plot_path = Path(args.plot_file)

    param_data = read_json(param_path)
    matches_data = read_json(matches_path)

    left_name = param_data["meta"]["left_image"]
    right_name = param_data["meta"]["right_image"]

    K1 = build_K(param_data["left"])
    K2 = build_K(param_data["right"])
    R = np.array(param_data["extrinsics"]["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(param_data["extrinsics"]["t"], dtype=np.float64).reshape(3)
    E, F = compute_E_F(K1, K2, R, t)

    pair, reverse_pair = find_target_pair(matches_data, left_name, right_name)
    metrics = compute_point_metrics(pair, reverse_pair, F)

    alg = metrics["algebraic_errors"]
    d1 = metrics["d1"]
    d2 = metrics["d2"]
    d_sym = metrics["d_sym"]
    singular_values = matrix_rank_singular_values(F)

    summary = {
        "num_matches": int(len(pair["matches"])),
        "invalid_line_count": int(metrics["invalid_line_count"]),
        "algebraic_error": {
            "mean": float(np.mean(alg)),
            "median": float(np.median(alg)),
            "abs_mean": float(np.mean(np.abs(alg))),
            "rms": float(np.sqrt(np.mean(alg ** 2))),
            "max_abs": float(np.max(np.abs(alg))),
        },
        "d1": summarize(d1),
        "d2": summarize(d2),
        "d_sym": summarize(d_sym),
        "F_singular_values": singular_values,
    }

    result = {
        "input": {
            "param_file": str(param_path),
            "matches_file": str(matches_path),
            "selected_pair": {
                "left_image": left_name,
                "right_image": right_name,
                "reversed_from_matches": reverse_pair,
            },
            "error_threshold": float(args.error_threshold),
        },
        "camera": {
            "K1": K1.tolist(),
            "K2": K2.tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
        },
        "summary": summary,
        "points": metrics["points"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_large_error_matches(param_path.parent, left_name, right_name, metrics["points"], args.error_threshold, plot_path)

    print("-" * 60)
    print(f"Selected pair      : {left_name} <-> {right_name}")
    print(f"Reversed from JSON : {reverse_pair}")
    print(f"Match count        : {summary['num_matches']}")
    print(f"Invalid lines      : {summary['invalid_line_count']}")
    print("R:")
    print(R)
    print(f"t: {t}")
    print("E:")
    print(E)
    print("F:")
    print(F)
    print(f"F singular values  : {singular_values}")
    print("Algebraic error:")
    print(f"  mean={summary['algebraic_error']['mean']:.6e} median={summary['algebraic_error']['median']:.6e} abs_mean={summary['algebraic_error']['abs_mean']:.6e} rms={summary['algebraic_error']['rms']:.6e} max_abs={summary['algebraic_error']['max_abs']:.6e}")
    print("Distances (pixels):")
    print(f"  d1   mean={summary['d1']['mean']:.6f} median={summary['d1']['median']:.6f} rms={summary['d1']['rms']:.6f} max={summary['d1']['max']:.6f}")
    print(f"  d2   mean={summary['d2']['mean']:.6f} median={summary['d2']['median']:.6f} rms={summary['d2']['rms']:.6f} max={summary['d2']['max']:.6f}")
    print(f"  dsym mean={summary['d_sym']['mean']:.6f} median={summary['d_sym']['median']:.6f} rms={summary['d_sym']['rms']:.6f} max={summary['d_sym']['max']:.6f}")
    print(f"Output JSON        : {output_path}")
    print(f"Output plot        : {plot_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()
