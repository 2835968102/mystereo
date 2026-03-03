#!/usr/bin/env python3
"""
SuperPoint stereo image matching.

Extracts keypoints and descriptors from stereo image pairs using the
SuperPoint network, then matches them with bidirectional nearest-neighbour
matching.  Output JSON is compatible with run_stereo_calib.

Usage:
    python superpoint_stereo_match.py \\
        --img_dir  /path/to/stereo_360 \\
        --weights  /path/to/superpoint_v1.pth \\
        --output   matches.json \\
        [--nn_thresh 0.7] [--conf_thresh 0.015] [--nms_dist 4] [--cuda]

The img_dir is expected to contain pairs named left_NNN.* / right_NNN.* and
optionally a camera_params.json with intrinsics + extrinsics initial values.
"""

import argparse
import itertools
import json
import os
import sys

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
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))

    dmat = np.sqrt(np.clip(2 - 2 * (desc1.T @ desc2), 0, None))   # N1×N2

    idx_12 = np.argmin(dmat, axis=1)   # best match in desc2 for each desc1
    idx_21 = np.argmin(dmat, axis=0)   # best match in desc1 for each desc2
    scores = dmat[np.arange(len(idx_12)), idx_12]

    mutual = (np.arange(len(idx_12)) == idx_21[idx_12])
    good   = (scores < nn_thresh) & mutual

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

def parse_args():
    p = argparse.ArgumentParser(description='SuperPoint stereo matching')
    p.add_argument('--img_dir',    required=True,
                   help='Directory with left_NNN.* / right_NNN.* images')
    p.add_argument('--weights',    required=True,
                   help='Path to superpoint_v1.pth')
    p.add_argument('--output',     default='superpoint_matches.json',
                   help='Output JSON path (default: superpoint_matches.json)')
    p.add_argument('--nn_thresh',   type=float, default=0.7,
                   help='Descriptor L2 distance threshold (default: 0.7)')
    p.add_argument('--conf_thresh', type=float, default=0.015,
                   help='Keypoint confidence threshold (default: 0.015)')
    p.add_argument('--nms_dist',    type=int,   default=4,
                   help='NMS suppression radius in pixels (default: 4)')
    p.add_argument('--cuda',        action='store_true',
                   help='Use CUDA if available')
    return p.parse_args()


def find_all_images(img_dir: str):
    """Return sorted list of all image file paths in img_dir."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    images = []
    for f in sorted(os.listdir(img_dir)):
        if os.path.splitext(f)[1].lower() in exts:
            images.append(os.path.join(img_dir, f))
    return images


def find_stereo_pairs(img_dir: str):
    """Return sorted list of (left_path, right_path, stem) tuples."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    left_files = {}
    for f in os.listdir(img_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in exts and name.startswith('left_'):
            stem = name[len('left_'):]
            left_files[stem] = os.path.join(img_dir, f)

    pairs = []
    for stem, lpath in sorted(left_files.items()):
        for ext in exts:
            rpath = os.path.join(img_dir, f'right_{stem}{ext}')
            if os.path.isfile(rpath):
                pairs.append((lpath, rpath, stem))
                break
    return pairs


def main():
    args = parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not use_cuda:
        print('Warning: CUDA requested but not available, falling back to CPU.')

    print(f'Loading SuperPoint weights from: {args.weights}')
    fe = SuperPointFrontend(
        weights_path=args.weights,
        nms_dist=args.nms_dist,
        conf_thresh=args.conf_thresh,
        nn_thresh=args.nn_thresh,
        cuda=use_cuda,
    )

    images = find_all_images(args.img_dir)
    if len(images) < 2:
        sys.exit(f'Need at least 2 images in: {args.img_dir}')
    n_pairs = len(images) * (len(images) - 1) // 2
    print(f'Found {len(images)} image(s), {n_pairs} pair(s) to match.')

    # Load optional camera params
    cam_json = os.path.join(args.img_dir, 'camera_params.json')
    if os.path.isfile(cam_json):
        with open(cam_json) as f:
            cam = json.load(f)
        result = {
            'left':       cam.get('left', {}),
            'right':      cam.get('right', {}),
            'extrinsics': cam.get('extrinsics', {}),
        }
        print(f'Loaded camera parameters from: {cam_json}')
    else:
        result = {}

    result['pairs'] = []

    # Cache descriptors to avoid recomputing per image
    cache = {}
    def get_feats(path):
        if path not in cache:
            cache[path] = fe.run(load_gray_float(path))
        return cache[path]

    total_matches = 0
    for lpath, rpath in itertools.combinations(images, 2):
        lname = os.path.basename(lpath)
        rname = os.path.basename(rpath)
        print(f'  {lname} <-> {rname} ...', end=' ', flush=True)

        pts_l, desc_l = get_feats(lpath)
        pts_r, desc_r = get_feats(rpath)

        matches = nn_match_two_way(desc_l, desc_r, args.nn_thresh)
        n_matches = matches.shape[1]
        total_matches += n_matches
        print(f'kpts L={pts_l.shape[1]}  R={pts_r.shape[1]}  matches={n_matches}')

        match_list = []
        for k in range(n_matches):
            score = float(matches[2, k])
            if score >= 0.4:
                continue
            i, j = int(matches[0, k]), int(matches[1, k])
            match_list.append({
                'left':  [round(float(pts_l[0, i]), 2),
                           round(float(pts_l[1, i]), 2)],
                'right': [round(float(pts_r[0, j]), 2),
                           round(float(pts_r[1, j]), 2)],
                'score': round(score, 4),
            })

        result['pairs'].append({
            'left_image':  lname,
            'right_image': rname,
            'num_keypoints': {'left': int(pts_l.shape[1]), 'right': int(pts_r.shape[1])},
            'num_matches': n_matches,
            'matches': match_list,
        })

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nDone. {total_matches} total matches across {n_pairs} pair(s).')
    print(f'Output written to: {args.output}')


if __name__ == '__main__':
    main()
