#include "offline_stereo_ba.h"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include <opencv2/calib3d.hpp>

#include "stereo_factors.h"

namespace stereocalib {
namespace {

struct BaselinePriorFactor {
  explicit BaselinePriorFactor(const std::vector<double>& init_extrinsics, double weight)
      : init_t_(3, 0.0), weight_(weight)
  {
    init_t_[0] = init_extrinsics[3];
    init_t_[1] = init_extrinsics[4];
    init_t_[2] = init_extrinsics[5];
  }

  bool operator()(const double* extrinsics, double* residual) const
  {
    residual[0] = weight_ * (extrinsics[3] - init_t_[0]);
    residual[1] = weight_ * (extrinsics[4] - init_t_[1]);
    residual[2] = weight_ * (extrinsics[5] - init_t_[2]);
    return true;
  }

  static ceres::CostFunction* Create(const std::vector<double>& init_extrinsics, double weight)
  {
    return new ceres::NumericDiffCostFunction<BaselinePriorFactor, ceres::CENTRAL, 3, 6>(
        new BaselinePriorFactor(init_extrinsics, weight));
  }

  std::vector<double> init_t_;
  double weight_ = 0.0;
};

struct AspectRatioPriorFactor {
  explicit AspectRatioPriorFactor(double weight) : weight_(weight) {}

  bool operator()(const double* intrinsics, double* residual) const
  {
    residual[0] = weight_ * (intrinsics[0] - intrinsics[1]);  // fx - fy
    return true;
  }

  static ceres::CostFunction* Create(double weight)
  {
    return new ceres::NumericDiffCostFunction<AspectRatioPriorFactor, ceres::CENTRAL, 1, 9>(
        new AspectRatioPriorFactor(weight));
  }

  double weight_ = 0.0;
};

struct TrackReprojFactor {
  TrackReprojFactor(const cv::Point2f& obs, bool is_left) : obs_(obs), is_left_(is_left) {}

  bool operator()(const double* intr_left,
                  const double* intr_right,
                  const double* extrinsics,
                  const double* frame_rvec,
                  const double* point3d,
                  double* residual) const
  {
    const cv::Mat rvec_lw = (cv::Mat_<double>(3, 1) << frame_rvec[0], frame_rvec[1], frame_rvec[2]);
    cv::Mat R_lw;
    cv::Rodrigues(rvec_lw, R_lw);

    const cv::Mat X_w = (cv::Mat_<double>(3, 1) << point3d[0], point3d[1], point3d[2]);
    const cv::Mat X_l = R_lw * X_w;

    cv::Mat X_cam = X_l;
    const double* intr = intr_left;

    if (!is_left_) {
      const cv::Mat rvec_rl = (cv::Mat_<double>(3, 1) << extrinsics[0], extrinsics[1], extrinsics[2]);
      cv::Mat R_rl;
      cv::Rodrigues(rvec_rl, R_rl);
      const cv::Mat t_rl = (cv::Mat_<double>(3, 1) << extrinsics[3], extrinsics[4], extrinsics[5]);
      X_cam = R_rl * X_l + t_rl;
      intr = intr_right;
    }

    const double Z = X_cam.at<double>(2, 0);
    if (Z <= 0.0) {
      residual[0] = 1e6;
      residual[1] = 1e6;
      return true;
    }

    const double xn = X_cam.at<double>(0, 0) / Z;
    const double yn = X_cam.at<double>(1, 0) / Z;

    double u = 0.0;
    double v = 0.0;
    ApplyDistAndProject(intr, xn, yn, u, v);

    residual[0] = static_cast<double>(obs_.x) - u;
    residual[1] = static_cast<double>(obs_.y) - v;
    return true;
  }

  static ceres::CostFunction* Create(const cv::Point2f& obs, bool is_left)
  {
    return new ceres::NumericDiffCostFunction<TrackReprojFactor, ceres::CENTRAL, 2, 9, 9, 6, 3, 3>(
        new TrackReprojFactor(obs, is_left));
  }

  cv::Point2f obs_;
  bool is_left_ = true;
};

cv::Mat ToRotation(const std::vector<double>& rvec)
{
  const cv::Mat rv = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]);
  cv::Mat R;
  cv::Rodrigues(rv, R);
  return R;
}

bool EstimatePureRotation(const std::vector<cv::Point2f>& pts_from,
                          const std::vector<cv::Point2f>& pts_to,
                          const cv::Mat& K,
                          const cv::Mat& dist,
                          cv::Mat& R_rel)
{
  if (pts_from.size() < 8 || pts_to.size() < 8) {
    return false;
  }

  std::vector<cv::Point2f> und_from;
  std::vector<cv::Point2f> und_to;
  cv::undistortPoints(pts_from, und_from, K, dist);
  cv::undistortPoints(pts_to, und_to, K, dist);

  cv::Mat inliers;
  const cv::Mat H = cv::findHomography(und_from, und_to, cv::RANSAC, 2.5, inliers, 2000, 0.995);
  if (H.empty()) {
    return false;
  }

  cv::SVD svd(H);
  cv::Mat R = svd.u * svd.vt;
  if (cv::determinant(R) < 0.0) {
    R = -R;
  }

  if (!cv::checkRange(R)) {
    return false;
  }

  R_rel = R;
  return true;
}

std::string Basename(const std::string& path)
{
  const size_t slash_pos = path.find_last_of("/\\");
  if (slash_pos == std::string::npos) {
    return path;
  }
  return path.substr(slash_pos + 1);
}

std::string PointKey(int image_idx, const cv::Point2f& pt)
{
  const int x100 = static_cast<int>(std::round(pt.x * 100.0f));
  const int y100 = static_cast<int>(std::round(pt.y * 100.0f));

  std::ostringstream oss;
  oss << image_idx << "_" << x100 << "_" << y100;
  return oss.str();
}

}  // namespace

int OfflineStereoBA::UnionFind::AddNode()
{
  const int idx = static_cast<int>(parent_.size());
  parent_.push_back(idx);
  rank_.push_back(0);
  return idx;
}

int OfflineStereoBA::UnionFind::Find(int x)
{
  if (parent_[x] != x) {
    parent_[x] = Find(parent_[x]);
  }
  return parent_[x];
}

void OfflineStereoBA::UnionFind::Unite(int a, int b)
{
  int pa = Find(a);
  int pb = Find(b);
  if (pa == pb) {
    return;
  }
  if (rank_[pa] < rank_[pb]) {
    std::swap(pa, pb);
  }
  parent_[pb] = pa;
  if (rank_[pa] == rank_[pb]) {
    rank_[pa]++;
  }
}

OfflineStereoBA::OfflineStereoBA(const OfflineBAInput& input, const Options& options)
    : input_(input), options_(options)
{
  intrinsics_left_ = input_.init_camera.left.ToVector();
  intrinsics_right_ = input_.init_camera.right.ToVector();
  extrinsics_ = input_.init_camera.extrinsics.ToVector();
  init_extrinsics_ = extrinsics_;
}

bool OfflineStereoBA::ParseImageName(const std::string& image_name, bool& is_left, std::string& frame_id) const
{
  const std::string base = Basename(image_name);
  const size_t dot = base.find_last_of('.');
  const std::string stem = dot == std::string::npos ? base : base.substr(0, dot);

  const std::string left_prefix = "left_";
  const std::string right_prefix = "right_";

  if (stem.compare(0, left_prefix.size(), left_prefix) == 0) {
    is_left = true;
    frame_id = stem.substr(left_prefix.size());
    return !frame_id.empty();
  }

  if (stem.compare(0, right_prefix.size(), right_prefix) == 0) {
    is_left = false;
    frame_id = stem.substr(right_prefix.size());
    return !frame_id.empty();
  }

  return false;
}

bool OfflineStereoBA::BuildTracks()
{
  std::unordered_map<std::string, int> image_index;
  images_.clear();

  auto ensure_image = [&](const std::string& name) {
    std::unordered_map<std::string, int>::const_iterator it = image_index.find(name);
    if (it != image_index.end()) {
      return it->second;
    }
    const int idx = static_cast<int>(images_.size());
    image_index[name] = idx;

    ImageInfo info;
    info.name = name;
    bool is_left = true;
    std::string frame_id;
    info.valid = ParseImageName(name, is_left, frame_id);
    info.is_left = is_left;
    info.frame_id = frame_id;
    images_.push_back(info);
    return idx;
  };

  struct NodeInfo {
    int image_idx = -1;
    cv::Point2f px;
  };

  std::vector<NodeInfo> nodes;
  nodes.reserve(200000);

  UnionFind uf;
  std::unordered_map<std::string, int> node_index;

  auto ensure_node = [&](int image_idx, const cv::Point2f& pt) {
    const std::string key = PointKey(image_idx, pt);
    std::unordered_map<std::string, int>::const_iterator it = node_index.find(key);
    if (it != node_index.end()) {
      return it->second;
    }
    const int node_id = uf.AddNode();
    node_index[key] = node_id;
    NodeInfo n;
    n.image_idx = image_idx;
    n.px = pt;
    nodes.push_back(n);
    return node_id;
  };

  for (size_t i = 0; i < input_.pairs.size(); ++i) {
    const RawImagePair& pair = input_.pairs[i];
    if (pair.image_a == pair.image_b) {
      continue;
    }

    const int ia = ensure_image(pair.image_a);
    const int ib = ensure_image(pair.image_b);

    std::vector<char> inlier_mask(pair.matches.size(), 1);
    bool a_left = true;
    bool b_left = true;
    std::string a_frame_id;
    std::string b_frame_id;
    const bool a_ok = ParseImageName(pair.image_a, a_left, a_frame_id);
    const bool b_ok = ParseImageName(pair.image_b, b_left, b_frame_id);

    std::vector<cv::Point2f> pts_a;
    std::vector<cv::Point2f> pts_b;
    std::vector<int> idx_map;
    pts_a.reserve(pair.matches.size());
    pts_b.reserve(pair.matches.size());
    idx_map.reserve(pair.matches.size());

    for (size_t k = 0; k < pair.matches.size(); ++k) {
      const RawPairMatch& m = pair.matches[k];
      if (m.score > options_.max_match_score) {
        continue;
      }
      pts_a.push_back(m.pt_a);
      pts_b.push_back(m.pt_b);
      idx_map.push_back(static_cast<int>(k));
    }

    bool pair_accepted = (pts_a.size() >= 8);
    if (pts_a.size() >= 8) {
      cv::Mat inliers;
      bool geometry_ok = false;

      if (a_ok && b_ok && a_left == b_left) {
        const cv::Mat H = cv::findHomography(pts_a, pts_b, cv::RANSAC, 3.0, inliers, 2000, 0.995);
        geometry_ok = !H.empty();
      } else {
        const cv::Mat F = cv::findFundamentalMat(pts_a, pts_b, cv::FM_RANSAC, 2.0, 0.995, inliers);
        geometry_ok = !F.empty();
      }

      const bool mask_row = (inliers.rows == static_cast<int>(pts_a.size()) && inliers.cols == 1);
      const bool mask_col = (inliers.cols == static_cast<int>(pts_a.size()) && inliers.rows == 1);
      if (geometry_ok && (mask_row || mask_col)) {
        std::fill(inlier_mask.begin(), inlier_mask.end(), 0);
        int inlier_count = 0;
        for (int k = 0; k < static_cast<int>(pts_a.size()); ++k) {
          const uchar ok = mask_row ? inliers.at<uchar>(k, 0) : inliers.at<uchar>(0, k);
          if (ok != 0) {
            inlier_mask[idx_map[k]] = 1;
            ++inlier_count;
          }
        }

        const double inlier_ratio = static_cast<double>(inlier_count) / static_cast<double>(pts_a.size());
        if (inlier_count < options_.min_pair_inliers || inlier_ratio < options_.min_pair_inlier_ratio) {
          pair_accepted = false;
        }
      } else {
        pair_accepted = false;
      }
    }

    if (!pair_accepted) {
      continue;
    }

    for (size_t k = 0; k < pair.matches.size(); ++k) {
      const RawPairMatch& m = pair.matches[k];
      if (m.score > options_.max_match_score) {
        continue;
      }
      if (!inlier_mask[k]) {
        continue;
      }
      const int na = ensure_node(ia, m.pt_a);
      const int nb = ensure_node(ib, m.pt_b);
      uf.Unite(na, nb);
    }
  }

  if (nodes.empty()) {
    std::cerr << "No valid match nodes were built." << std::endl;
    return false;
  }

  // Build frame list from parsed image names.
  std::map<std::string, int> frame_map;
  for (size_t i = 0; i < images_.size(); ++i) {
    ImageInfo& img = images_[i];
    if (!img.valid) {
      continue;
    }

    std::map<std::string, int>::const_iterator it = frame_map.find(img.frame_id);
    if (it == frame_map.end()) {
      const int fidx = static_cast<int>(frames_.size());
      frame_map[img.frame_id] = fidx;
      FrameState frame;
      frame.frame_id = img.frame_id;
      frames_.push_back(frame);
      frame_ids_.push_back(img.frame_id);
      img.frame_idx = fidx;
    } else {
      img.frame_idx = it->second;
    }

    FrameState& f = frames_[img.frame_idx];
    if (img.is_left) {
      f.left_image_idx = static_cast<int>(i);
    } else {
      f.right_image_idx = static_cast<int>(i);
    }
  }

  std::vector<FrameState> valid_frames;
  std::vector<std::string> valid_frame_ids;
  std::vector<int> old_to_new_frame(frames_.size(), -1);

  for (size_t i = 0; i < frames_.size(); ++i) {
    const FrameState& f = frames_[i];
    if (f.left_image_idx < 0 || f.right_image_idx < 0) {
      continue;
    }
    old_to_new_frame[i] = static_cast<int>(valid_frames.size());
    valid_frames.push_back(f);
    valid_frame_ids.push_back(f.frame_id);
  }

  frames_.swap(valid_frames);
  frame_ids_.swap(valid_frame_ids);

  for (size_t i = 0; i < images_.size(); ++i) {
    ImageInfo& img = images_[i];
    if (img.frame_idx >= 0) {
      img.frame_idx = old_to_new_frame[img.frame_idx];
    }
  }

  if (frames_.empty()) {
    std::cerr << "No valid stereo frame (left_xxx + right_xxx) found." << std::endl;
    return false;
  }

  std::unordered_map<int, std::vector<int> > comps;
  comps.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int root = uf.Find(static_cast<int>(i));
    comps[root].push_back(static_cast<int>(i));
  }

  tracks_.clear();
  num_observations_ = 0;

  for (std::unordered_map<int, std::vector<int> >::const_iterator it = comps.begin(); it != comps.end(); ++it) {
    const std::vector<int>& comp = it->second;
    Track track;
    std::set<std::pair<int, bool> > seen;

    for (size_t j = 0; j < comp.size(); ++j) {
      const NodeInfo& node = nodes[comp[j]];
      const ImageInfo& img = images_[node.image_idx];
      if (!img.valid || img.frame_idx < 0) {
        continue;
      }

      const std::pair<int, bool> key(img.frame_idx, img.is_left);
      if (seen.find(key) != seen.end()) {
        continue;
      }
      seen.insert(key);

      TrackObservation obs;
      obs.frame_idx = img.frame_idx;
      obs.is_left = img.is_left;
      obs.px = node.px;
      track.observations.push_back(obs);
    }

    if (track.observations.size() < static_cast<size_t>(options_.min_track_len)) {
      continue;
    }

    bool has_left = false;
    bool has_right = false;
    for (size_t j = 0; j < track.observations.size(); ++j) {
      has_left = has_left || track.observations[j].is_left;
      has_right = has_right || !track.observations[j].is_left;
    }
    if (!has_left || !has_right) {
      continue;
    }

    track.point3d.assign(3, 0.0);
    tracks_.push_back(track);
    num_observations_ += track.observations.size();
  }

  num_tracks_ = tracks_.size();

  if (tracks_.empty()) {
    std::cerr << "No valid tracks after union-find filtering." << std::endl;
    return false;
  }

  return true;
}

size_t OfflineStereoBA::CollectLeftLeftCorrespondences(int frame_a, int frame_b,
                                                       std::vector<cv::Point2f>& pts_a,
                                                       std::vector<cv::Point2f>& pts_b) const
{
  pts_a.clear();
  pts_b.clear();

  for (size_t t = 0; t < tracks_.size(); ++t) {
    const Track& track = tracks_[t];
    const TrackObservation* obs_a = NULL;
    const TrackObservation* obs_b = NULL;

    for (size_t k = 0; k < track.observations.size(); ++k) {
      const TrackObservation& obs = track.observations[k];
      if (!obs.is_left) {
        continue;
      }
      if (obs.frame_idx == frame_a) {
        obs_a = &obs;
      } else if (obs.frame_idx == frame_b) {
        obs_b = &obs;
      }
      if (obs_a != NULL && obs_b != NULL) {
        break;
      }
    }

    if (obs_a != NULL && obs_b != NULL) {
      pts_a.push_back(obs_a->px);
      pts_b.push_back(obs_b->px);
    }
  }

  return pts_a.size();
}

bool OfflineStereoBA::InitializeFrameRotations(std::vector<int>& registration_order)
{
  if (frames_.empty()) {
    return false;
  }
  registration_order.clear();

  const cv::Mat K = input_.init_camera.left.K();
  const cv::Mat dist = input_.init_camera.left.dist();

  // Pick the start frame with highest number of left observations.
  int start_frame = 0;
  size_t best_count = 0;
  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    size_t cnt = 0;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      const Track& track = tracks_[ti];
      for (size_t oi = 0; oi < track.observations.size(); ++oi) {
        const TrackObservation& obs = track.observations[oi];
        if (obs.frame_idx == static_cast<int>(fi) && obs.is_left) {
          cnt++;
          break;
        }
      }
    }
    if (cnt > best_count) {
      best_count = cnt;
      start_frame = static_cast<int>(fi);
    }
  }

  frames_[start_frame].initialized = true;
  frames_[start_frame].rvec.assign(3, 0.0);
  fixed_frame_idx_ = start_frame;
  registration_order.push_back(start_frame);

  size_t registered = 1;
  std::vector<cv::Point2f> pts_from;
  std::vector<cv::Point2f> pts_to;

  while (registered < frames_.size()) {
    int best_from = -1;
    int best_to = -1;
    size_t max_overlap = 0;

    for (size_t i = 0; i < frames_.size(); ++i) {
      if (!frames_[i].initialized) {
        continue;
      }
      for (size_t j = 0; j < frames_.size(); ++j) {
        if (frames_[j].initialized) {
          continue;
        }

        const size_t overlap = CollectLeftLeftCorrespondences(static_cast<int>(i), static_cast<int>(j), pts_from, pts_to);
        if (overlap > max_overlap) {
          max_overlap = overlap;
          best_from = static_cast<int>(i);
          best_to = static_cast<int>(j);
        }
      }
    }

    if (best_to < 0 || max_overlap < 8) {
      // Fall back to identity for the remaining frames.
      for (size_t i = 0; i < frames_.size(); ++i) {
        if (!frames_[i].initialized) {
          frames_[i].initialized = true;
          frames_[i].rvec.assign(3, 0.0);
          registration_order.push_back(static_cast<int>(i));
          registered++;
        }
      }
      break;
    }

    CollectLeftLeftCorrespondences(best_from, best_to, pts_from, pts_to);

    cv::Mat R_rel;
    if (!EstimatePureRotation(pts_from, pts_to, K, dist, R_rel)) {
      frames_[best_to].initialized = true;
      frames_[best_to].rvec = frames_[best_from].rvec;
      registration_order.push_back(best_to);
      registered++;
      continue;
    }

    const cv::Mat R_from = ToRotation(frames_[best_from].rvec);
    const cv::Mat R_to = R_rel * R_from;

    cv::Mat rvec_to;
    cv::Rodrigues(R_to, rvec_to);
    frames_[best_to].rvec = {
        rvec_to.at<double>(0, 0),
        rvec_to.at<double>(1, 0),
        rvec_to.at<double>(2, 0),
    };
    frames_[best_to].initialized = true;
    registration_order.push_back(best_to);
    registered++;
  }

  return true;
}

bool OfflineStereoBA::InitializeTrackPoints()
{
  const cv::Mat K_l = input_.init_camera.left.K();
  const cv::Mat dist_l = input_.init_camera.left.dist();
  const cv::Mat K_r = input_.init_camera.right.K();
  const cv::Mat dist_r = input_.init_camera.right.dist();

  StereoExtrinsics ext = input_.init_camera.extrinsics;
  if (ext.R.empty() || ext.t.empty()) {
    ext.FromVector(extrinsics_);
  }

  cv::Mat P_l = cv::Mat::eye(3, 4, CV_64F);
  cv::Mat P_r(3, 4, CV_64F);
  ext.R.copyTo(P_r(cv::Range(0, 3), cv::Range(0, 3)));
  ext.t.copyTo(P_r(cv::Range(0, 3), cv::Range(3, 4)));

  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    Track& track = tracks_[ti];
    bool initialized = false;

    // Preferred init: triangulate from a same-frame stereo observation.
    for (size_t oi = 0; oi < track.observations.size() && !initialized; ++oi) {
      const TrackObservation& obs_l = track.observations[oi];
      if (!obs_l.is_left) {
        continue;
      }

      for (size_t oj = 0; oj < track.observations.size(); ++oj) {
        const TrackObservation& obs_r = track.observations[oj];
        if (obs_r.is_left || obs_r.frame_idx != obs_l.frame_idx) {
          continue;
        }

        std::vector<cv::Point2f> p_l(1, obs_l.px);
        std::vector<cv::Point2f> p_r(1, obs_r.px);
        std::vector<cv::Point2f> p_l_n;
        std::vector<cv::Point2f> p_r_n;
        cv::undistortPoints(p_l, p_l_n, K_l, dist_l);
        cv::undistortPoints(p_r, p_r_n, K_r, dist_r);

        cv::Mat X4;
        cv::triangulatePoints(P_l, P_r, p_l_n, p_r_n, X4);

        const double w = X4.at<float>(3, 0);
        if (std::abs(w) < 1e-9) {
          continue;
        }

        const cv::Mat Xl = (cv::Mat_<double>(3, 1)
            << X4.at<float>(0, 0) / w,
               X4.at<float>(1, 0) / w,
               X4.at<float>(2, 0) / w);

        if (Xl.at<double>(2, 0) <= 0.0) {
          continue;
        }

        const cv::Mat R_lw = ToRotation(frames_[obs_l.frame_idx].rvec);
        const cv::Mat Xw = R_lw.t() * Xl;

        track.point3d[0] = Xw.at<double>(0, 0);
        track.point3d[1] = Xw.at<double>(1, 0);
        track.point3d[2] = Xw.at<double>(2, 0);
        initialized = true;
        break;
      }
    }

    if (initialized) {
      continue;
    }

    // Fallback init from any left observation with fixed depth.
    for (size_t oi = 0; oi < track.observations.size() && !initialized; ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (!obs.is_left) {
        continue;
      }

      std::vector<cv::Point2f> p(1, obs.px);
      std::vector<cv::Point2f> p_n;
      cv::undistortPoints(p, p_n, K_l, dist_l);

      const cv::Mat ray_l = (cv::Mat_<double>(3, 1)
          << p_n[0].x,
             p_n[0].y,
             1.0);

      const cv::Mat R_lw = ToRotation(frames_[obs.frame_idx].rvec);
      const cv::Mat Xw = R_lw.t() * (5.0 * ray_l);
      track.point3d[0] = Xw.at<double>(0, 0);
      track.point3d[1] = Xw.at<double>(1, 0);
      track.point3d[2] = Xw.at<double>(2, 0);
      initialized = true;
    }

    if (!initialized) {
      track.point3d[0] = 0.0;
      track.point3d[1] = 0.0;
      track.point3d[2] = 5.0;
    }
  }

  return true;
}

bool OfflineStereoBA::RunBundleAdjustment(const std::vector<char>& active_frames,
                                          int max_num_iterations,
                                          ceres::Solver::Summary& summary,
                                          double& init_rmse,
                                          double& final_rmse,
                                          int frame_to_optimize)
{
  ceres::Problem problem;
  size_t active_residuals = 0;

  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    const Track& track = tracks_[ti];

    int active_obs = 0;
    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames_.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) || !active_frames[obs.frame_idx]) {
        continue;
      }
      active_obs++;
    }

    if (active_obs < 2) {
      continue;
    }

    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames_.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) || !active_frames[obs.frame_idx]) {
        continue;
      }

      ceres::CostFunction* cost = TrackReprojFactor::Create(obs.px, obs.is_left);
      ceres::LossFunction* loss = new ceres::HuberLoss(options_.huber_delta);
      problem.AddResidualBlock(cost,
                               loss,
                               intrinsics_left_.data(),
                               intrinsics_right_.data(),
                               extrinsics_.data(),
                               frames_[obs.frame_idx].rvec.data(),
                               tracks_[ti].point3d.data());
      active_residuals++;
    }
  }

  if (active_residuals == 0) {
    return false;
  }

  if (options_.baseline_prior_weight > 0.0) {
    ceres::CostFunction* prior_cost = BaselinePriorFactor::Create(init_extrinsics_, options_.baseline_prior_weight);
    problem.AddResidualBlock(prior_cost, NULL, extrinsics_.data());
  }
  if (options_.aspect_ratio_prior_weight > 0.0) {
    ceres::CostFunction* aspect_left = AspectRatioPriorFactor::Create(options_.aspect_ratio_prior_weight);
    ceres::CostFunction* aspect_right = AspectRatioPriorFactor::Create(options_.aspect_ratio_prior_weight);
    problem.AddResidualBlock(aspect_left, NULL, intrinsics_left_.data());
    problem.AddResidualBlock(aspect_right, NULL, intrinsics_right_.data());
  }

  std::vector<int> fixed_intrinsic_indices;
  fixed_intrinsic_indices.push_back(2);
  fixed_intrinsic_indices.push_back(3);
  if (options_.fix_distortion) {
    fixed_intrinsic_indices.push_back(4);
    fixed_intrinsic_indices.push_back(5);
    fixed_intrinsic_indices.push_back(6);
    fixed_intrinsic_indices.push_back(7);
    fixed_intrinsic_indices.push_back(8);
  }

  auto set_intrinsics_bounds = [&](double* intr, const Intrinsics& init_intr) {
    problem.SetParameterLowerBound(intr, 0, 0.5 * init_intr.fx);
    problem.SetParameterUpperBound(intr, 0, 1.5 * init_intr.fx);
    problem.SetParameterLowerBound(intr, 1, 0.5 * init_intr.fy);
    problem.SetParameterUpperBound(intr, 1, 1.5 * init_intr.fy);
    problem.SetParameterLowerBound(intr, 4, -1.0);
    problem.SetParameterUpperBound(intr, 4, 1.0);
    problem.SetParameterLowerBound(intr, 5, -1.0);
    problem.SetParameterUpperBound(intr, 5, 1.0);
    problem.SetParameterLowerBound(intr, 6, -0.2);
    problem.SetParameterUpperBound(intr, 6, 0.2);
    problem.SetParameterLowerBound(intr, 7, -0.2);
    problem.SetParameterUpperBound(intr, 7, 0.2);
    problem.SetParameterLowerBound(intr, 8, -1.0);
    problem.SetParameterUpperBound(intr, 8, 1.0);
  };

  if (problem.HasParameterBlock(intrinsics_left_.data())) {
    problem.SetManifold(intrinsics_left_.data(), new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(intrinsics_left_.data(), input_.init_camera.left);
  }
  if (problem.HasParameterBlock(intrinsics_right_.data())) {
    problem.SetManifold(intrinsics_right_.data(), new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(intrinsics_right_.data(), input_.init_camera.right);
  }

  // 固定帧的位姿（根据优化模式决定哪些帧需要固定）
  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    if (!problem.HasParameterBlock(frames_[fi].rvec.data())) {
      continue;
    }

    bool should_fix = false;

    // 模式1：增量BA模式 - 只优化指定的新注册帧
    if (frame_to_optimize >= 0) {
      // 固定除了新注册帧之外的所有帧
      should_fix = (static_cast<int>(fi) != frame_to_optimize);
    }
    // 模式2：全局BA模式 - 优化所有已注册帧（除参考帧外）
    else {
      // 固定参考帧和未激活的帧
      should_fix = (static_cast<int>(fi) == fixed_frame_idx_ ||
                    fi >= active_frames.size() ||
                    !active_frames[fi]);
    }

    if (should_fix) {
      problem.SetParameterBlockConstant(frames_[fi].rvec.data());
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = std::max(1, max_num_iterations);
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 4;
  options.minimizer_progress_to_stdout = true;

  ceres::Solve(options, &problem, &summary);

  if (summary.num_residuals <= 0) {
    init_rmse = 0.0;
    final_rmse = 0.0;
  } else {
    init_rmse = std::sqrt(2.0 * summary.initial_cost / summary.num_residuals);
    final_rmse = std::sqrt(2.0 * summary.final_cost / summary.num_residuals);
  }

  return true;
}

void OfflineStereoBA::ApplyResult(StereoCamera& result)
{
  result.left.FromVector(intrinsics_left_);
  result.right.FromVector(intrinsics_right_);
  result.extrinsics.FromVector(extrinsics_);
}

void OfflineStereoBA::SetGroundTruth(const StereoCamera& gt)
{
  has_ground_truth_ = true;
  ground_truth_ = gt;
}

void OfflineStereoBA::PrintCurrentVsGroundTruth(const std::string& stage_name) const
{
  if (!has_ground_truth_) {
    return;
  }

  // 获取当前优化结果
  StereoCamera current;
  current.left.FromVector(intrinsics_left_);
  current.right.FromVector(intrinsics_right_);
  current.extrinsics.FromVector(extrinsics_);

  std::cout << "\n========== " << stage_name << " - Comparison with Ground Truth ==========\n";
  std::cout << std::showpos << std::fixed << std::setprecision(6);

  // 左相机内参对比
  std::cout << "Left camera:\n";
  std::cout << "  fx: " << current.left.fx << " (gt: " << ground_truth_.left.fx
            << ", diff: " << (current.left.fx - ground_truth_.left.fx) << ")\n";
  std::cout << "  fy: " << current.left.fy << " (gt: " << ground_truth_.left.fy
            << ", diff: " << (current.left.fy - ground_truth_.left.fy) << ")\n";
  std::cout << "  cx: " << current.left.cx << " (gt: " << ground_truth_.left.cx
            << ", diff: " << (current.left.cx - ground_truth_.left.cx) << ")\n";
  std::cout << "  cy: " << current.left.cy << " (gt: " << ground_truth_.left.cy
            << ", diff: " << (current.left.cy - ground_truth_.left.cy) << ")\n";

  // 右相机内参对比
  std::cout << "Right camera:\n";
  std::cout << "  fx: " << current.right.fx << " (gt: " << ground_truth_.right.fx
            << ", diff: " << (current.right.fx - ground_truth_.right.fx) << ")\n";
  std::cout << "  fy: " << current.right.fy << " (gt: " << ground_truth_.right.fy
            << ", diff: " << (current.right.fy - ground_truth_.right.fy) << ")\n";
  std::cout << "  cx: " << current.right.cx << " (gt: " << ground_truth_.right.cx
            << ", diff: " << (current.right.cx - ground_truth_.right.cx) << ")\n";
  std::cout << "  cy: " << current.right.cy << " (gt: " << ground_truth_.right.cy
            << ", diff: " << (current.right.cy - ground_truth_.right.cy) << ")\n";

  // 外参对比 - 平移向量
  const double tx_diff = current.extrinsics.t.at<double>(0, 0) - ground_truth_.extrinsics.t.at<double>(0, 0);
  const double ty_diff = current.extrinsics.t.at<double>(1, 0) - ground_truth_.extrinsics.t.at<double>(1, 0);
  const double tz_diff = current.extrinsics.t.at<double>(2, 0) - ground_truth_.extrinsics.t.at<double>(2, 0);

  std::cout << "Extrinsics:\n";
  std::cout << "  t: [" << current.extrinsics.t.at<double>(0, 0) << ", "
            << current.extrinsics.t.at<double>(1, 0) << ", "
            << current.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  gt: [" << ground_truth_.extrinsics.t.at<double>(0, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(1, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  diff: [" << tx_diff << ", " << ty_diff << ", " << tz_diff << "]\n";

  // 基线距离对比
  const double baseline_current = std::sqrt(
      current.extrinsics.t.at<double>(0, 0) * current.extrinsics.t.at<double>(0, 0) +
      current.extrinsics.t.at<double>(1, 0) * current.extrinsics.t.at<double>(1, 0) +
      current.extrinsics.t.at<double>(2, 0) * current.extrinsics.t.at<double>(2, 0));
  const double baseline_gt = std::sqrt(
      ground_truth_.extrinsics.t.at<double>(0, 0) * ground_truth_.extrinsics.t.at<double>(0, 0) +
      ground_truth_.extrinsics.t.at<double>(1, 0) * ground_truth_.extrinsics.t.at<double>(1, 0) +
      ground_truth_.extrinsics.t.at<double>(2, 0) * ground_truth_.extrinsics.t.at<double>(2, 0));

  std::cout << "  baseline: " << baseline_current << " (gt: " << baseline_gt
            << ", diff: " << (baseline_current - baseline_gt) << ")\n";

  // 旋转误差
  cv::Mat R_diff = current.extrinsics.R * ground_truth_.extrinsics.R.t();
  const double tr = R_diff.at<double>(0, 0) + R_diff.at<double>(1, 1) + R_diff.at<double>(2, 2);
  double cos_theta = (tr - 1.0) * 0.5;
  cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
  const double rot_error_deg = std::acos(cos_theta) * 57.2957795130823208768;

  std::cout << "  rotation_error: " << rot_error_deg << " degrees\n";
  std::cout << std::noshowpos;
  std::cout << "================================================================\n\n";
}

bool OfflineStereoBA::Solve(StereoCamera& result)
{
  if (!BuildTracks()) {
    return false;
  }

  std::vector<int> registration_order;
  if (!InitializeFrameRotations(registration_order)) {
    return false;
  }

  if (!InitializeTrackPoints()) {
    return false;
  }

  if (registration_order.empty()) {
    return false;
  }

  std::vector<char> active_frames(frames_.size(), 0);
  active_frames[registration_order[0]] = 1;

  bool have_rmse = false;
  int successful_registrations = 0;

  for (size_t i = 1; i < registration_order.size(); ++i) {
    const int frame_idx = registration_order[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(active_frames.size())) {
      continue;
    }
    active_frames[frame_idx] = 1;
    successful_registrations++;

    // 增量BA：只优化当前新注册的帧
    ceres::Solver::Summary incremental_summary;
    double step_init_rmse = 0.0;
    double step_final_rmse = 0.0;
    if (RunBundleAdjustment(active_frames,
                            options_.incremental_max_iter,
                            incremental_summary,
                            step_init_rmse,
                            step_final_rmse,
                            frame_idx)) {  // 传入当前新注册的帧索引
      if (!have_rmse) {
        init_reproj_error_ = step_init_rmse;
        have_rmse = true;
      }
      final_reproj_error_ = step_final_rmse;
      std::cout << "[Incremental BA] registered_frames=" << (i + 1)
                << "/" << registration_order.size()
                << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                << step_final_rmse << " px" << std::endl;

      // 输出当前结果与真值的对比
      PrintCurrentVsGroundTruth("Incremental BA - Frame " + std::to_string(i + 1));
    }

    if (options_.global_opt_interval > 0 &&
        successful_registrations % options_.global_opt_interval == 0) {
      // 周期性全局BA：优化所有已注册帧（除参考帧外）
      ceres::Solver::Summary periodic_summary;
      double global_init_rmse = 0.0;
      double global_final_rmse = 0.0;
      if (RunBundleAdjustment(active_frames,
                              options_.max_iter,
                              periodic_summary,
                              global_init_rmse,
                              global_final_rmse)) {  // 未传入frame_to_optimize，使用默认值-1（全局BA模式）
        if (!have_rmse) {
          init_reproj_error_ = global_init_rmse;
          have_rmse = true;
        }
        final_reproj_error_ = global_final_rmse;
        std::cout << "[Global BA] registered_frames=" << (i + 1)
                  << "/" << registration_order.size()
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << global_final_rmse << " px" << std::endl;

        // 输出当前结果与真值的对比
        PrintCurrentVsGroundTruth("Periodic Global BA - Frame " + std::to_string(i + 1));
      }
    }
  }

  // 最终全局BA：优化所有帧（除参考帧外）
  std::fill(active_frames.begin(), active_frames.end(), 1);
  if (!RunBundleAdjustment(active_frames,
                           options_.max_iter,
                           summary_,
                           init_reproj_error_,
                           final_reproj_error_)) {  // 未传入frame_to_optimize，使用默认值-1（全局BA模式）
    return false;
  }

  std::cout << summary_.BriefReport() << std::endl;
  std::cout << "Tracks=" << num_tracks_ << ", observations=" << num_observations_ << ", frames=" << frames_.size()
            << std::endl;
  std::cout << "Reprojection error: init=" << std::fixed << std::setprecision(4) << init_reproj_error_
            << " px, final=" << final_reproj_error_ << " px" << std::endl;

  // 输出最终优化结果与真值的对比
  PrintCurrentVsGroundTruth("Final Global BA");

  const bool converged = (summary_.termination_type == ceres::CONVERGENCE ||
                          summary_.termination_type == ceres::NO_CONVERGENCE);
  const bool pass_reproj = (final_reproj_error_ <= options_.max_reproj_error);

  if (!converged) {
    std::cerr << "Offline BA did not converge." << std::endl;
  }
  if (!pass_reproj) {
    std::cerr << "Final reprojection error " << final_reproj_error_
              << " px exceeds threshold " << options_.max_reproj_error << " px." << std::endl;
  }

  ApplyResult(result);
  return converged && pass_reproj;
}

}  // namespace stereocalib
