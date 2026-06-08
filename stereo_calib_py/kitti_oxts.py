"""Utilities for deriving lightweight KITTI RAW frame pose priors from OXTS."""

from __future__ import annotations

import json
import math
from pathlib import Path


EARTH_RADIUS_M = 6378137.0


def _mercator_xy(lat_deg: float, lon_deg: float, scale: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = scale * lon_rad * EARTH_RADIUS_M
    y = scale * EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))
    return x, y


def _read_oxts_lat_lon_alt(path: Path) -> tuple[float, float, float]:
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 3:
        raise ValueError(f"Invalid KITTI OXTS record: {path}")
    return float(values[0]), float(values[1]), float(values[2])


def _read_oxts_pose(path: Path) -> tuple[float, float, float, float, float, float]:
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 6:
        raise ValueError(f"Invalid KITTI OXTS record: {path}")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def _read_kitti_calib_values(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if not parts:
                continue
            try:
                values[key.strip()] = [float(part) for part in parts]
            except ValueError:
                continue
    return values


def _reshape_3x3(values: list[float]) -> list[list[float]]:
    if len(values) != 9:
        raise ValueError("Expected 9 values for a 3x3 matrix")
    return [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ]


def _matmul_3x3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[row][0] * b[0][col] +
            a[row][1] * b[1][col] +
            a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _matvec_3x3(a: list[list[float]], v: list[float]) -> list[float]:
    return [
        a[row][0] * v[0] + a[row][1] * v[1] + a[row][2] * v[2]
        for row in range(3)
    ]


def _transpose_3x3(a: list[list[float]]) -> list[list[float]]:
    return [[a[col][row] for col in range(3)] for row in range(3)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vec_neg(a: list[float]) -> list[float]:
    return [-a[0], -a[1], -a[2]]


def _rotation_matrix_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> list[list[float]]:
    """KITTI OXTS roll/pitch/yaw to rotation matrix, using Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return _matmul_3x3(_matmul_3x3(rz, ry), rx)


def _rotation_vector_from_matrix(r: list[list[float]]) -> list[float]:
    """Convert a 3x3 rotation matrix to an OpenCV-style Rodrigues vector."""
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0]

    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-8:
        # This is only a fallback for near-180-degree absolute orientations.
        # KITTI frame-to-frame relative angles are small, but the absolute yaw
        # can sit close to pi. Pick an axis from the diagonal in that case.
        xx = max(0.0, (r[0][0] + 1.0) * 0.5)
        yy = max(0.0, (r[1][1] + 1.0) * 0.5)
        zz = max(0.0, (r[2][2] + 1.0) * 0.5)
        axis = [math.sqrt(xx), math.sqrt(yy), math.sqrt(zz)]
        if r[0][1] < 0.0:
            axis[1] = -axis[1]
        if r[0][2] < 0.0:
            axis[2] = -axis[2]
    else:
        axis = [
            (r[2][1] - r[1][2]) / (2.0 * sin_angle),
            (r[0][2] - r[2][0]) / (2.0 * sin_angle),
            (r[1][0] - r[0][1]) / (2.0 * sin_angle),
        ]
    return [angle * axis[0], angle * axis[1], angle * axis[2]]


def _rotation_vector_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    return _rotation_vector_from_matrix(_rotation_matrix_from_rpy(roll, pitch, yaw))


def _kitti_raw_calib_dir(drive_dir: Path) -> Path:
    date_dir = drive_dir.parent
    return date_dir.parent / f"{date_dir.name}_calib" / date_dir.name


def _load_rectified_cam0_from_imu_transform(
    drive_dir: Path,
) -> tuple[list[list[float]], list[float]] | None:
    """Return rectified cam0-from-IMU R,t without reading KITTI P_rect intrinsics."""
    calib_dir = _kitti_raw_calib_dir(drive_dir)
    imu_to_velo_file = calib_dir / "calib_imu_to_velo.txt"
    velo_to_cam_file = calib_dir / "calib_velo_to_cam.txt"
    cam_to_cam_file = calib_dir / "calib_cam_to_cam.txt"
    if not (imu_to_velo_file.exists() and velo_to_cam_file.exists() and cam_to_cam_file.exists()):
        return None

    imu_to_velo = _read_kitti_calib_values(imu_to_velo_file)
    velo_to_cam = _read_kitti_calib_values(velo_to_cam_file)
    cam_to_cam = _read_kitti_calib_values(cam_to_cam_file)
    if (
        "R" not in imu_to_velo
        or "T" not in imu_to_velo
        or "R" not in velo_to_cam
        or "T" not in velo_to_cam
        or "R_rect_00" not in cam_to_cam
    ):
        return None

    r_velo_imu = _reshape_3x3(imu_to_velo["R"])
    t_velo_imu = imu_to_velo["T"][:3]
    r_cam_velo = _reshape_3x3(velo_to_cam["R"])
    t_cam_velo = velo_to_cam["T"][:3]
    r_rect_cam = _reshape_3x3(cam_to_cam["R_rect_00"])
    r_cam_imu = _matmul_3x3(_matmul_3x3(r_rect_cam, r_cam_velo), r_velo_imu)
    t_cam_imu_unrect = _vec_add(_matvec_3x3(r_cam_velo, t_velo_imu), t_cam_velo)
    t_cam_imu = _matvec_3x3(r_rect_cam, t_cam_imu_unrect)
    return r_cam_imu, t_cam_imu


def kitti_raw_oxts_total_distance(left_img_dir: Path) -> float:
    """Return total OXTS path length for KITTI RAW frames matching left images."""
    drive_dir = left_img_dir.parent.parent
    oxts_dir = drive_dir / "oxts" / "data"
    if not oxts_dir.exists():
        raise FileNotFoundError(f"KITTI OXTS directory not found: {oxts_dir}")

    image_files = sorted(left_img_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No KITTI left images found: {left_img_dir}")

    records: list[tuple[float, float, float]] = []
    for image_path in image_files:
        oxts_path = oxts_dir / f"{image_path.stem}.txt"
        if not oxts_path.exists():
            continue
        records.append(_read_oxts_lat_lon_alt(oxts_path))

    if len(records) < 2:
        return 0.0

    origin_lat, origin_lon, origin_alt = records[0]
    scale = math.cos(math.radians(origin_lat))
    origin_x, origin_y = _mercator_xy(origin_lat, origin_lon, scale)

    positions: list[tuple[float, float, float]] = []
    for lat, lon, alt in records:
        x, y = _mercator_xy(lat, lon, scale)
        positions.append((x - origin_x, y - origin_y, alt - origin_alt))

    total = 0.0
    for prev, cur in zip(positions, positions[1:]):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        dz = cur[2] - prev[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def build_kitti_raw_oxts_frame_poses(left_img_dir: Path, output_path: Path) -> Path:
    """Write frame positions from KITTI RAW OXTS files in the pipeline JSON format."""
    drive_dir = left_img_dir.parent.parent
    oxts_dir = drive_dir / "oxts" / "data"
    if not oxts_dir.exists():
        raise FileNotFoundError(f"KITTI OXTS directory not found: {oxts_dir}")

    image_files = sorted(left_img_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No KITTI left images found: {left_img_dir}")

    records: list[tuple[str, float, float, float, float, float, float]] = []
    missing = []
    for image_path in image_files:
        frame_id = image_path.stem
        oxts_path = oxts_dir / f"{frame_id}.txt"
        if not oxts_path.exists():
            missing.append(oxts_path)
            continue
        lat, lon, alt, roll, pitch, yaw = _read_oxts_pose(oxts_path)
        records.append((frame_id, lat, lon, alt, roll, pitch, yaw))

    if not records:
        raise FileNotFoundError(f"No matching KITTI OXTS records found in: {oxts_dir}")
    if missing:
        missing_preview = ", ".join(str(path.name) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} KITTI OXTS records, first missing: {missing_preview}"
        )

    origin_lat, origin_lon, origin_alt = records[0][1:4]
    scale = math.cos(math.radians(origin_lat))
    origin_x, origin_y = _mercator_xy(origin_lat, origin_lon, scale)
    cam_imu_transform = _load_rectified_cam0_from_imu_transform(drive_dir)
    r_cam_imu = cam_imu_transform[0] if cam_imu_transform is not None else None
    t_cam_imu = cam_imu_transform[1] if cam_imu_transform is not None else None
    cam_center_imu = None
    if r_cam_imu is not None and t_cam_imu is not None:
        cam_center_imu = _matvec_3x3(_transpose_3x3(r_cam_imu), _vec_neg(t_cam_imu))

    frames = []
    for frame_id, lat, lon, alt, roll, pitch, yaw in records:
        x, y = _mercator_xy(lat, lon, scale)
        imu_position = [x - origin_x, y - origin_y, alt - origin_alt]
        r_world_imu = _rotation_matrix_from_rpy(roll, pitch, yaw)
        if r_cam_imu is not None:
            # BA frame poses use world-to-left-camera rotations. KITTI OXTS gives
            # IMU-to-world orientation; this rotation chain maps world vectors
            # into the rectified left camera frame without using P_rect intrinsics.
            r_cam_world = _matmul_3x3(r_cam_imu, _transpose_3x3(r_world_imu))
            if cam_center_imu is not None:
                position = _vec_add(
                    imu_position,
                    _matvec_3x3(r_world_imu, cam_center_imu),
                )
            else:
                position = imu_position
            rotation = _rotation_vector_from_matrix(r_cam_world)
            rotation_frame = "rectified_cam0_world_to_camera"
        else:
            position = imu_position
            rotation = _rotation_vector_from_matrix(r_world_imu)
            rotation_frame = "oxts_imu_to_world"
        frames.append(
            {
                "frame_id": frame_id,
                "position": position,
                "translation": position,
                "imu_position": imu_position,
                "rotation": rotation,
                "rotation_frame": rotation_frame,
                "rotation_rpy": [roll, pitch, yaw],
            }
        )

    payload = {
        "source": "kitti_raw_oxts",
        "drive_dir": str(drive_dir),
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return output_path
