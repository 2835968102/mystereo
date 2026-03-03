'''
Evaluate stereo calibration results.

Usage:
    python eval_stereo.py --pred result.json --gt ground_truth.json

Both files follow the same JSON schema as the optimizer output:
  { "left": {...}, "right": {...}, "extrinsics": {"R": [...], "t": [...]} }

Metrics reported
----------------
  focal_error_left/right   – absolute difference in focal length (pixels)
  R_error                  – rotation angle between predicted and GT R_rl (degrees)
  t_error                  – magnitude of translation error, as fraction of GT baseline
'''

import json
import math
import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate stereo calibration results')
    parser.add_argument('--pred', required=True, help='Path to prediction JSON')
    parser.add_argument('--gt',   required=True, help='Path to ground-truth JSON')
    return parser.parse_args()


# ─── Intrinsic error ─────────────────────────────────────────────────────────

def focal_error(pred_intr: dict, gt_intr: dict) -> float:
    '''Mean absolute error of fx and fy (pixels).'''
    err_fx = abs(pred_intr['fx'] - gt_intr['fx'])
    err_fy = abs(pred_intr['fy'] - gt_intr['fy'])
    return 0.5 * (err_fx + err_fy)


def distortion_error(pred_intr: dict, gt_intr: dict) -> float:
    '''L2 norm of the distortion coefficient vector difference.'''
    keys = ['k1', 'k2', 'p1', 'p2', 'k3']
    diff = np.array([pred_intr.get(k, 0.0) - gt_intr.get(k, 0.0) for k in keys])
    return float(np.linalg.norm(diff))


# ─── Extrinsic error ─────────────────────────────────────────────────────────

def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    '''
    Angular distance between two rotation matrices (degrees).
    Uses the formula:  angle = arccos( (trace(R_rel) - 1) / 2 )
    '''
    R_rel = R_pred @ R_gt.T
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    '''
    Absolute translation error normalised by the GT baseline magnitude.
    Returns NaN if the GT baseline is zero.
    '''
    baseline = np.linalg.norm(t_gt)
    if baseline < 1e-9:
        return float('nan')
    return float(np.linalg.norm(t_pred - t_gt) / baseline)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pred = read_json(args.pred)
    gt   = read_json(args.gt)

    # Intrinsic errors
    fl_err_l = focal_error(pred['left'],  gt['left'])
    fl_err_r = focal_error(pred['right'], gt['right'])
    dt_err_l = distortion_error(pred['left'],  gt['left'])
    dt_err_r = distortion_error(pred['right'], gt['right'])

    # Extrinsic errors
    R_pred = np.array(pred['extrinsics']['R']).reshape(3, 3)
    t_pred = np.array(pred['extrinsics']['t']).reshape(3)
    R_gt   = np.array(gt['extrinsics']['R']).reshape(3, 3)
    t_gt   = np.array(gt['extrinsics']['t']).reshape(3)

    rot_err = rotation_error_deg(R_pred, R_gt)
    tra_err = translation_error(t_pred, t_gt)

    print('─' * 50)
    print(f'Focal error  left : {fl_err_l:.4f} px')
    print(f'Focal error  right: {fl_err_r:.4f} px')
    print(f'Distortion   left : {dt_err_l:.6f}')
    print(f'Distortion   right: {dt_err_r:.6f}')
    print(f'Rotation error    : {rot_err:.4f} deg')
    print(f'Translation error : {tra_err:.4f} (relative to baseline)')
    print('─' * 50)


if __name__ == '__main__':
    main()
