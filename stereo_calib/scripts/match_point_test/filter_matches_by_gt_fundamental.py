import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_MATCHES_DIR = Path("/home/hello/pml/mycalib/stereo_calib/result/match_points")
DEFAULT_BLENDER_DIR = Path("/home/hello/pml/mycalib/blender-file")
DEFAULT_OUTPUT_DIR = DEFAULT_MATCHES_DIR
EPS = 1e-12


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)



def parse_args():
    parser = argparse.ArgumentParser(description="Filter matches using pair-specific GT fundamental matrices")
    parser.add_argument("--matches_dir", default=str(DEFAULT_MATCHES_DIR), help="Directory containing input matches JSON files")
    parser.add_argument("--blender_dir", default=str(DEFAULT_BLENDER_DIR), help="Directory containing stereo_panoramic_xx camera files")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for filtered matches JSON files")
    parser.add_argument("--stats_dir", default=None, help="Directory for stats JSON files (default: output_dir)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Keep matches with d_sym <= threshold")
    parser.add_argument("--metric", choices=["dsym"], default="dsym", help="Filtering metric")
    parser.add_argument("--dry_run", action="store_true", help="Only compute and print stats")
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



def compute_fundamental(K1: np.ndarray, K2: np.ndarray, R21: np.ndarray, t21: np.ndarray) -> np.ndarray:
    E = skew(t21) @ R21
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)



def safe_point_line_distance(line: np.ndarray, point_h: np.ndarray) -> float:
    denom = math.hypot(float(line[0]), float(line[1]))
    if denom < EPS:
        return float("nan")
    return abs(float(point_h @ line)) / denom



def parse_image_info(image_name: str) -> tuple[str, str]:
    base = Path(image_name).name
    stem = base.rsplit(".", 1)[0]
    if stem.startswith("left_"):
        return "left", stem[len("left_"):]
    if stem.startswith("right_"):
        return "right", stem[len("right_"):]
    raise ValueError(f"Unsupported image naming convention: {image_name}")



def load_pose_lookup(poses_data: dict) -> dict:
    lookup = {}
    for frame in poses_data["frames"]:
        left_image = frame["left_image"]
        right_image = frame["right_image"]
        lookup[left_image] = {
            "side": "left",
            "frame": frame["index"],
            "R": np.array(frame["left_pose"]["R"], dtype=np.float64).reshape(3, 3),
            "t": np.array(frame["left_pose"]["t"], dtype=np.float64).reshape(3),
        }
        lookup[right_image] = {
            "side": "right",
            "frame": frame["index"],
            "R": np.array(frame["right_pose"]["R"], dtype=np.float64).reshape(3, 3),
            "t": np.array(frame["right_pose"]["t"], dtype=np.float64).reshape(3),
        }
    return lookup



def classify_pair_type(img1: str, img2: str) -> str:
    side1, frame1 = parse_image_info(img1)
    side2, frame2 = parse_image_info(img2)
    if frame1 == frame2 and side1 != side2:
        return "stereo"
    if side1 == "left" and side2 == "left":
        return "LL"
    if side1 == "right" and side2 == "right":
        return "RR"
    if side1 == "left" and side2 == "right":
        return "LR"
    return "RL"



def summarize_counts(pairs: list[dict], thresholds: list[int]) -> dict:
    counts = {str(v): 0 for v in thresholds}
    for pair in pairs:
        n = int(pair["num_matches"])
        for th in thresholds:
            if n >= th:
                counts[str(th)] += 1
    return counts



def summarize_left_left_init_candidates(pairs: list[dict], thresholds: list[int]) -> dict:
    counts = {str(v): 0 for v in thresholds}
    for pair in pairs:
        if classify_pair_type(pair["left_image"], pair["right_image"]) != "LL":
            continue
        n = int(pair["num_matches"])
        for th in thresholds:
            if n >= th:
                counts[str(th)] += 1
    return counts



def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def pair_specific_f(pair: dict, pose_lookup: dict, K_left: np.ndarray, K_right: np.ndarray):
    img1 = pair["left_image"]
    img2 = pair["right_image"]
    pose1 = pose_lookup[img1]
    pose2 = pose_lookup[img2]
    K1 = K_left if pose1["side"] == "left" else K_right
    K2 = K_left if pose2["side"] == "left" else K_right
    R21 = pose2["R"] @ pose1["R"].T
    t21 = pose2["t"] - R21 @ pose1["t"]
    F21 = compute_fundamental(K1, K2, R21, t21)
    return F21, R21, t21



def filter_pair(pair: dict, F21: np.ndarray, threshold: float) -> tuple[dict, dict]:
    kept_matches = []
    dsym_values = []
    invalid_count = 0

    for match in pair["matches"]:
        x1 = np.array([match["left"][0], match["left"][1], 1.0], dtype=np.float64)
        x2 = np.array([match["right"][0], match["right"][1], 1.0], dtype=np.float64)
        l2 = F21 @ x1
        l1 = F21.T @ x2
        d2 = safe_point_line_distance(l2, x2)
        d1 = safe_point_line_distance(l1, x1)
        d_sym = d1 + d2 if not (math.isnan(d1) or math.isnan(d2)) else float("nan")
        dsym_values.append(d_sym)
        if math.isnan(d_sym):
            invalid_count += 1
            continue
        if d_sym <= threshold:
            kept_matches.append(match)

    new_pair = {
        "left_image": pair["left_image"],
        "right_image": pair["right_image"],
        "num_keypoints": pair.get("num_keypoints", {}),
        "num_matches": len(kept_matches),
        "matches": kept_matches,
    }

    valid_dsym = [v for v in dsym_values if not math.isnan(v)]
    pair_stats = {
        "left_image": pair["left_image"],
        "right_image": pair["right_image"],
        "pair_type": classify_pair_type(pair["left_image"], pair["right_image"]),
        "original_matches": len(pair["matches"]),
        "kept_matches": len(kept_matches),
        "removed_matches": len(pair["matches"]) - len(kept_matches),
        "invalid_matches": invalid_count,
        "retention_ratio": float(len(kept_matches) / len(pair["matches"])) if pair["matches"] else 1.0,
        "d_sym_median_before": float(np.median(valid_dsym)) if valid_dsym else float("nan"),
        "d_sym_max_before": float(np.max(valid_dsym)) if valid_dsym else float("nan"),
    }
    return new_pair, pair_stats



def build_stats(original_pairs: list[dict], filtered_pairs: list[dict], pair_stats: list[dict], threshold: float) -> dict:
    thresholds = [8, 12, 50]
    original_total = sum(len(pair["matches"]) for pair in original_pairs)
    kept_total = sum(int(pair["num_matches"]) for pair in filtered_pairs)

    type_stats = {}
    for pair_type in ["stereo", "LL", "RR", "LR", "RL"]:
        rows = [row for row in pair_stats if row["pair_type"] == pair_type]
        orig = sum(row["original_matches"] for row in rows)
        kept = sum(row["kept_matches"] for row in rows)
        type_stats[pair_type] = {
            "pair_count": len(rows),
            "original_matches": orig,
            "kept_matches": kept,
            "retention_ratio": float(kept / orig) if orig else 1.0,
        }

    dropped_below_50 = [
        row for row in pair_stats
        if row["original_matches"] >= 50 and row["kept_matches"] < 50
    ]
    ll_dropped_below_12 = [
        row for row in pair_stats
        if row["pair_type"] == "LL" and row["original_matches"] >= 12 and row["kept_matches"] < 12
    ]

    worst_pairs = sorted(pair_stats, key=lambda row: (row["retention_ratio"], row["kept_matches"]))[:20]

    return {
        "filter": {
            "metric": "dsym",
            "threshold": float(threshold),
        },
        "global": {
            "pair_count": len(original_pairs),
            "original_match_count": original_total,
            "kept_match_count": kept_total,
            "removed_match_count": original_total - kept_total,
            "retention_ratio": float(kept_total / original_total) if original_total else 1.0,
        },
        "pair_type_stats": type_stats,
        "pair_threshold_counts": {
            "before": summarize_counts(original_pairs, thresholds),
            "after": summarize_counts(filtered_pairs, thresholds),
        },
        "left_left_init_candidate_counts": {
            "before": summarize_left_left_init_candidates(original_pairs, thresholds),
            "after": summarize_left_left_init_candidates(filtered_pairs, thresholds),
        },
        "dropped_below_50": dropped_below_50,
        "left_left_dropped_below_12": ll_dropped_below_12,
        "worst_retention_pairs": worst_pairs,
        "pairs": pair_stats,
    }



def print_summary(stats: dict, label: str):
    print("-" * 72)
    print(f"Dataset                     : {label}")
    print(f"Metric / threshold          : {stats['filter']['metric']} <= {stats['filter']['threshold']}")
    print(f"Total matches               : {stats['global']['original_match_count']} -> {stats['global']['kept_match_count']} ({stats['global']['retention_ratio']:.3f})")
    print(f"Pairs with >=50 matches     : {stats['pair_threshold_counts']['before']['50']} -> {stats['pair_threshold_counts']['after']['50']}")
    print(f"LL pairs with >=12 matches  : {stats['left_left_init_candidate_counts']['before']['12']} -> {stats['left_left_init_candidate_counts']['after']['12']}")
    print(f"Pairs dropped below 50      : {len(stats['dropped_below_50'])}")
    print(f"LL pairs dropped below 12   : {len(stats['left_left_dropped_below_12'])}")
    print("Pair-type retention:")
    for pair_type, row in stats["pair_type_stats"].items():
        print(f"  {pair_type:>6}: pairs={row['pair_count']:4d} matches={row['original_matches']:6d} -> {row['kept_matches']:6d} ({row['retention_ratio']:.3f})")
    print("Worst-retention examples:")
    for row in stats["worst_retention_pairs"][:10]:
        print(
            f"  {row['left_image']} -> {row['right_image']}: "
            f"{row['original_matches']} -> {row['kept_matches']} ({row['retention_ratio']:.3f}), "
            f"type={row['pair_type']}"
        )
    print("-" * 72)



def infer_dataset_name(matches_path: Path) -> str:
    name = matches_path.stem
    suffix = "_matches"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected matches filename: {matches_path.name}")
    return name



def infer_dataset_id(dataset_name: str) -> str:
    suffix = "_matches"
    if not dataset_name.endswith(suffix):
        raise ValueError(f"Unexpected dataset name: {dataset_name}")
    return dataset_name[len("panoramic_"):-len(suffix)]



def format_threshold_suffix(threshold: float) -> str:
    return format(threshold, "g")



def filter_single_file(matches_path: Path, blender_dir: Path, output_dir: Path, stats_dir: Path, threshold: float, dry_run: bool):
    dataset_name = infer_dataset_name(matches_path)
    dataset_id = infer_dataset_id(dataset_name)
    camera_path = blender_dir / f"stereo_panoramic_{dataset_id}" / "camera_params.json"
    poses_path = blender_dir / f"stereo_panoramic_{dataset_id}" / "camera_poses_summary.json"
    threshold_suffix = format_threshold_suffix(threshold)
    output_path = output_dir / f"{dataset_name}_gtpose_dsym_le_{threshold_suffix}px.json"
    stats_path = stats_dir / f"{dataset_name}_gtpose_dsym_le_{threshold_suffix}px.stats.json"

    matches_data = read_json(matches_path)
    camera_data = read_json(camera_path)
    poses_data = read_json(poses_path)

    K_left = build_K(camera_data["left"])
    K_right = build_K(camera_data["right"])
    pose_lookup = load_pose_lookup(poses_data)

    original_pairs = matches_data["pairs"]
    filtered_pairs = []
    pair_stats = []

    for pair in original_pairs:
        F21, _, _ = pair_specific_f(pair, pose_lookup, K_left, K_right)
        filtered_pair, stats_row = filter_pair(pair, F21, threshold)
        filtered_pairs.append(filtered_pair)
        pair_stats.append(stats_row)

    stats = build_stats(original_pairs, filtered_pairs, pair_stats, threshold)
    print_summary(stats, dataset_name)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(stats), f, indent=2)

    if dry_run:
        print(f"Dry run only. Stats JSON: {stats_path}")
        return

    result = {
        "left": matches_data["left"],
        "right": matches_data["right"],
        "extrinsics": matches_data["extrinsics"],
        "pairs": filtered_pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Output matches JSON         : {output_path}")
    print(f"Output stats JSON           : {stats_path}")



def main():
    args = parse_args()
    matches_dir = Path(args.matches_dir)
    blender_dir = Path(args.blender_dir)
    output_dir = Path(args.output_dir)
    stats_dir = Path(args.stats_dir) if args.stats_dir else output_dir

    matches_files = sorted(
        path for path in matches_dir.glob("panoramic_*_matches.json")
        if "_gtpose_dsym_le_" not in path.name
    )
    if not matches_files:
        raise FileNotFoundError(f"No original matches JSON files found in {matches_dir}")

    for matches_path in matches_files:
        filter_single_file(matches_path, blender_dir, output_dir, stats_dir, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
