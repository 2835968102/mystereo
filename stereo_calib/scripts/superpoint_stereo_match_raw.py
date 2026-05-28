#!/usr/bin/env python3
"""
SuperPoint stereo image matching.

Extracts keypoints and descriptors from stereo image pairs using the
SuperPoint network, then matches them with bidirectional nearest-neighbour
matching.  Output JSON is compatible with run_stereo_calib.

Usage:
    python superpoint_stereo_match.py \\
        --left_img_dir  /path/to/image_00/data \\
        --right_img_dir /path/to/image_01/data \\
        --output        matches.json \\
        [--nn_thresh 0.7] [--conf_thresh 0.015] [--nms_dist 4] [--cuda]

The left/right image roots are scanned directly and the script outputs match data only.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SuperPoint network definition (self-contained, no external import needed)
# ---------------------------------------------------------------------------

class SuperPointNet(nn.Module):
    """Pytorch SuperPoint network."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared encoder
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        # Detector head
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, d1, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc / (dn + 1e-8)
        return semi, desc


# ---------------------------------------------------------------------------
# SuperPoint frontend: NMS, keypoint extraction, descriptor interpolation
# ---------------------------------------------------------------------------

class SuperPointFrontend:
    CELL = 8
    BORDER = 4

    def __init__(self, weights_path: str, nms_dist: int, conf_thresh: float,
                 nn_thresh: float, cuda: bool = False):
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cuda = cuda

        self.net = SuperPointNet()
        map_loc = 'cuda' if cuda else 'cpu'
        state = torch.load(weights_path, map_location=map_loc)
        self.net.load_state_dict(state)
        if cuda:
            self.net = self.net.cuda()
        self.net.eval()

    # -- NMS -----------------------------------------------------------------

    def _nms_fast(self, corners: np.ndarray, H: int, W: int) -> np.ndarray:
        """Fast NMS on 3×N corner array [x, y, conf]."""
        grid = np.zeros((H, W), dtype=int)
        inds = np.zeros((H, W), dtype=int)
        order = np.argsort(-corners[2])
        corners = corners[:, order]
        rc = corners[:2].round().astype(int)
        if rc.shape[1] == 0:
            return np.zeros((3, 0))
        if rc.shape[1] == 1:
            return np.vstack((rc, corners[2])).reshape(3, 1)
        for i in range(rc.shape[1]):
            grid[rc[1, i], rc[0, i]] = 1
            inds[rc[1, i], rc[0, i]] = i
        pad = self.nms_dist
        grid = np.pad(grid, pad, mode='constant')
        for i in range(rc.shape[1]):
            px, py = rc[0, i] + pad, rc[1, i] + pad
            if grid[py, px] == 1:
                grid[py - pad:py + pad + 1, px - pad:px + pad + 1] = 0
                grid[py, px] = -1
        ky, kx = np.where(grid == -1)
        ky, kx = ky - pad, kx - pad
        kept_inds = inds[ky, kx]
        out = corners[:, kept_inds]
        out = out[:, np.argsort(-out[2])]
        return out

    # -- Run -----------------------------------------------------------------

    def run(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : H×W float32 in [0, 1], grayscale.

        Returns
        -------
        pts  : 3×N  [x, y, conf]
        desc : 256×N unit-norm descriptors
        """
        assert img.ndim == 2 and img.dtype == np.float32
        H, W = img.shape
        inp = torch.from_numpy(img).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()

        with torch.no_grad():
            semi, coarse_desc = self.net(inp)

        # --- keypoints ---
        semi = semi.squeeze().cpu().numpy()
        dense = np.exp(semi)
        dense /= dense.sum(axis=0) + 1e-5
        nodust = dense[:-1].transpose(1, 2, 0)
        Hc, Wc = H // self.CELL, W // self.CELL
        heatmap = nodust.reshape(Hc, Wc, self.CELL, self.CELL)
        heatmap = heatmap.transpose(0, 2, 1, 3).reshape(Hc * self.CELL, Wc * self.CELL)

        ys, xs = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), np.zeros((256, 0))

        pts = np.stack([xs.astype(float), ys.astype(float), heatmap[ys, xs]])
        pts = self._nms_fast(pts, H, W)

        b = self.BORDER
        mask = ((pts[0] >= b) & (pts[0] < W - b) &
                (pts[1] >= b) & (pts[1] < H - b))
        pts = pts[:, mask]

        # --- descriptors ---
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            return pts, np.zeros((D, 0))

        samp = torch.from_numpy(pts[:2].copy()).float()
        samp[0] = samp[0] / (W / 2.0) - 1.0
        samp[1] = samp[1] / (H / 2.0) - 1.0
        samp = samp.T.contiguous().view(1, 1, -1, 2)
        if self.cuda:
            samp = samp.cuda()
        with torch.no_grad():
            desc = F.grid_sample(coarse_desc, samp, align_corners=True)
        desc = desc[0, :, 0, :].cpu().numpy()         # D×N
        norms = np.linalg.norm(desc, axis=0, keepdims=True)
        desc /= norms + 1e-8
        return pts, desc


# ---------------------------------------------------------------------------
# Bidirectional nearest-neighbour matching
# ---------------------------------------------------------------------------

def nn_match_two_way(desc1: np.ndarray, desc2: np.ndarray,
                     nn_thresh: float) -> np.ndarray:
    """
    Parameters
    ----------
    desc1, desc2 : D×N1, D×N2 unit-norm descriptor arrays.
    nn_thresh    : L2 distance threshold.

    Returns
    -------
    matches : 3×M  [idx1, idx2, distance]
    """
    if desc1.shape[1] == 0 or desc2.shape[1] < 2:
        return np.zeros((3, 0))

    dmat = np.sqrt(np.clip(2 - 2 * (desc1.T @ desc2), 0, None))   # N1×N2

    idx_12 = np.argmin(dmat, axis=1)   # best match in desc2 for each desc1
    idx_21 = np.argmin(dmat, axis=0)   # best match in desc1 for each desc2
    scores = dmat[np.arange(len(idx_12)), idx_12]

    # Lowe ratio / margin test:
    # 1) best match must be clearly better than the second-best candidate;
    # 2) this removes ambiguous matches in repeated texture regions.
    #
    # We use only the two smallest distances per row to keep the extra cost low.
    second_scores = np.partition(dmat, 1, axis=1)[:, 1]
    ratio_thresh = 0.80
    margin_thresh = 0.05

    mutual = (np.arange(len(idx_12)) == idx_21[idx_12])
    ratio_ok = scores / (second_scores + 1e-8) < ratio_thresh
    margin_ok = (second_scores - scores) > margin_thresh
    good = (scores < nn_thresh) & mutual & ratio_ok & margin_ok

    m_idx1 = np.where(good)[0]
    m_idx2 = idx_12[good]
    matches = np.zeros((3, good.sum()))
    matches[0] = m_idx1
    matches[1] = m_idx2
    matches[2] = scores[good]
    return matches


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_gray_float(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIXED_WEIGHTS_PATH = Path(__file__).resolve().parent / 'superpoint_v1.pth'


def parse_args():
    p = argparse.ArgumentParser(description='SuperPoint stereo matching')
    p.add_argument('--left_img_dir', required=True,
                   help='Left image root directory, such as image_00/data')
    p.add_argument('--right_img_dir', required=True,
                   help='Right image root directory, such as image_01/data')
    p.add_argument('--output', default='superpoint_matches.json',
                   help='Output JSON path (default: superpoint_matches.json)')
    p.add_argument('--nn_thresh', type=float, default=0.7,
                   help='Descriptor L2 distance threshold (default: 0.7)')
    p.add_argument('--conf_thresh', type=float, default=0.015,
                   help='Keypoint confidence threshold (default: 0.015)')
    p.add_argument('--nms_dist', type=int, default=4,
                   help='NMS suppression radius in pixels (default: 4)')
    p.add_argument('--cuda', action='store_true',
                   help='Use CUDA if available')
    return p.parse_args()


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')


def build_image_records(image_dir: str, image_label: str, is_left: bool):
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'{image_label}目录不存在: {root}')

    records = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        records.append({
            'path': str(path),
            'image_name': f'{root.parent.name}/{path.name}',
            'frame_id': path.stem,
            'is_left': is_left,
        })
    return records


def build_sparse_kitti_pairs(left_records, right_records):
    left_by_frame = {record['frame_id']: record for record in left_records}
    right_by_frame = {record['frame_id']: record for record in right_records}
    frame_ids = sorted(set(left_by_frame) | set(right_by_frame))

    missing_left = [frame_id for frame_id in frame_ids if frame_id not in left_by_frame]
    missing_right = [frame_id for frame_id in frame_ids if frame_id not in right_by_frame]
    if missing_left or missing_right:
        parts = []
        if missing_left:
            parts.append(f"missing left frames: {', '.join(missing_left[:5])}")
        if missing_right:
            parts.append(f"missing right frames: {', '.join(missing_right[:5])}")
        raise ValueError('Incomplete stereo sequence: ' + '; '.join(parts))

    pair_records = []
    for idx, frame_id in enumerate(frame_ids):
        pair_records.append((left_by_frame[frame_id], right_by_frame[frame_id]))
        if idx + 1 >= len(frame_ids):
            continue
        next_frame_id = frame_ids[idx + 1]
        pair_records.append((left_by_frame[frame_id], left_by_frame[next_frame_id]))
        pair_records.append((right_by_frame[frame_id], right_by_frame[next_frame_id]))

    return pair_records


def main():
    args = parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not use_cuda:
        print('Warning: CUDA requested but not available, falling back to CPU.')

    print(f'Loading SuperPoint weights from: {FIXED_WEIGHTS_PATH}')
    fe = SuperPointFrontend(
        weights_path=FIXED_WEIGHTS_PATH,
        nms_dist=args.nms_dist,
        conf_thresh=args.conf_thresh,
        nn_thresh=args.nn_thresh,
        cuda=use_cuda,
    )

    try:
        left_records = build_image_records(args.left_img_dir, '左图像根', True)
        right_records = build_image_records(args.right_img_dir, '右图像根', False)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    image_records = left_records + right_records
    if len(image_records) < 2:
        sys.exit(f'Need at least 2 images in dataset selection: {args.left_img_dir} {args.right_img_dir}')

    try:
        pair_records = build_sparse_kitti_pairs(left_records, right_records)
    except ValueError as exc:
        sys.exit(str(exc))

    n_pairs = len(pair_records)
    print(f'Found {len(image_records)} image(s), {n_pairs} sparse pair(s) to match.')

    result = {'pairs': []}

    # Cache descriptors to avoid recomputing per image
    cache = {}

    def get_feats(path):
        if path not in cache:
            cache[path] = fe.run(load_gray_float(path))
        return cache[path]

    total_matches = 0
    for rec_a, rec_b in pair_records:
        print(f"  {rec_a['image_name']} <-> {rec_b['image_name']} ...", end=' ', flush=True)

        pts_a, desc_a = get_feats(rec_a['path'])
        pts_b, desc_b = get_feats(rec_b['path'])

        matches = nn_match_two_way(desc_a, desc_b, args.nn_thresh)
        n_matches = matches.shape[1]
        total_matches += n_matches
        print(f'kpts A={pts_a.shape[1]}  B={pts_b.shape[1]}  matches={n_matches}')

        match_list = []
        for k in range(n_matches):
            score = float(matches[2, k])
            i, j = int(matches[0, k]), int(matches[1, k])
            match_list.append({
                'left': [round(float(pts_a[0, i]), 2),
                         round(float(pts_a[1, i]), 2)],
                'right': [round(float(pts_b[0, j]), 2),
                          round(float(pts_b[1, j]), 2)],
                'score': round(score, 4),
            })

        result['pairs'].append({
            'left_image': rec_a['image_name'],
            'right_image': rec_b['image_name'],
            'image_a': rec_a['image_name'],
            'image_b': rec_b['image_name'],
            'image_a_frame_id': rec_a['frame_id'],
            'image_b_frame_id': rec_b['frame_id'],
            'image_a_is_left': rec_a['is_left'],
            'image_b_is_left': rec_b['is_left'],
            'num_keypoints': {'left': int(pts_a.shape[1]), 'right': int(pts_b.shape[1])},
            'num_matches': n_matches,
            'matches': match_list,
        })

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nDone. {total_matches} total matches across {n_pairs} pair(s).')
    print(f'Output written to: {args.output}')


if __name__ == '__main__':
    main()
