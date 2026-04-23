#include "track_builder.h"


#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include <opencv2/calib3d.hpp>

#include "core/camera_math.h"
#include "frame_score_utils.h"

namespace stereocalib {
namespace {


cv::Mat ToColumnMat(const std::vector<double>& vec)
{
  return (cv::Mat_<double>(3, 1) << vec[0], vec[1], vec[2]);
}

cv::Mat WorldFromCamera(const FrameState& frame, const cv::Mat& X_l)
{
  const cv::Mat R_lw = camera_math::ToRotation(frame.rvec);
  const cv::Mat t_lw = ToColumnMat(frame.tvec);
  return R_lw.t() * (X_l - t_lw);
}

bool EstimatePoseFromEssential(const std::vector<cv::Point2f>& pts_from,
                               const std::vector<cv::Point2f>& pts_to,
                               const cv::Mat& K,
                               const cv::Mat& dist,
                               cv::Mat& rvec_to,
                               cv::Mat& tvec_to)
{
  if (pts_from.size() < 8 || pts_to.size() < 8) {
    return false;
  }

  std::vector<cv::Point2f> und_from;
  std::vector<cv::Point2f> und_to;
  cv::undistortPoints(pts_from, und_from, K, dist);
  cv::undistortPoints(pts_to, und_to, K, dist);

  cv::Mat inliers;
  const cv::Mat E = cv::findEssentialMat(und_from, und_to, 1.0, cv::Point2d(0.0, 0.0),
                                         cv::RANSAC, 0.999, 1e-3, inliers);
  if (E.empty()) {
    return false;
  }

  cv::Mat R_rel;
  cv::Mat t_rel;
  const int recovered = cv::recoverPose(E, und_from, und_to, R_rel, t_rel, 1.0, cv::Point2d(0.0, 0.0), inliers);
  if (recovered < 8) {
    return false;
  }

  cv::Rodrigues(R_rel, rvec_to);
  tvec_to = t_rel.clone();
  return cv::checkRange(R_rel) && cv::checkRange(tvec_to);
}

bool CollectStereoPnPData(const StereoCamera& init_camera,
                          const std::vector<Track>& tracks,
                          const std::vector<FrameState>& frames,
                          int source_frame_idx,
                          int target_frame_idx,
                          std::vector<cv::Point3f>& object_points,
                          std::vector<cv::Point2f>& image_points)
{
  object_points.clear();
  image_points.clear();

  const cv::Mat K_l = init_camera.left.K();
  const cv::Mat dist_l = init_camera.left.dist();
  const cv::Mat K_r = init_camera.right.K();
  const cv::Mat dist_r = init_camera.right.dist();
  const cv::Mat R_rl = init_camera.extrinsics.R;
  const cv::Mat t_rl = init_camera.extrinsics.t;

  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    const Track& track = tracks[ti];
    const TrackObservation* src_left = nullptr;
    const TrackObservation* src_right = nullptr;
    const TrackObservation* target_left = nullptr;

    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.frame_idx == source_frame_idx) {
        if (obs.is_left) {
          src_left = &obs;
        } else {
          src_right = &obs;
        }
      } else if (obs.frame_idx == target_frame_idx && obs.is_left) {
        target_left = &obs;
      }
    }

    if (src_left == nullptr || src_right == nullptr || target_left == nullptr) {
      continue;
    }

    cv::Point3d X_l;
    if (!camera_math::TriangulatePoint(src_left->px, src_right->px,
                                       K_l, K_r, dist_l, dist_r, R_rl, t_rl, X_l)) {
      continue;
    }
    if (X_l.z <= 0.0) {
      continue;
    }

    const cv::Mat Xw = WorldFromCamera(frames[source_frame_idx],
                                       (cv::Mat_<double>(3, 1) << X_l.x, X_l.y, X_l.z));
    object_points.emplace_back(static_cast<float>(Xw.at<double>(0, 0)),
                               static_cast<float>(Xw.at<double>(1, 0)),
                               static_cast<float>(Xw.at<double>(2, 0)));
    image_points.push_back(target_left->px);
  }

  return object_points.size() >= 6;
}

bool EstimatePoseFromPnP(const std::vector<cv::Point3f>& object_points,
                         const std::vector<cv::Point2f>& image_points,
                         const cv::Mat& K,
                         const cv::Mat& dist,
                         const FrameState& initial_frame,
                         cv::Mat& rvec,
                         cv::Mat& tvec,
                         int& inlier_count)
{
  inlier_count = 0;
  if (object_points.size() < 6 || image_points.size() < 6 || object_points.size() != image_points.size()) {
    return false;
  }

  rvec = ToColumnMat(initial_frame.rvec);
  tvec = ToColumnMat(initial_frame.tvec);
  cv::Mat inliers;
  const bool ok = cv::solvePnPRansac(object_points,
                                     image_points,
                                     K,
                                     dist,
                                     rvec,
                                     tvec,
                                     true,
                                     200,
                                     8.0,
                                     0.99,
                                     inliers,
                                     cv::SOLVEPNP_ITERATIVE);
  if (!ok) {
    return false;
  }

  inlier_count = inliers.rows;
  return inlier_count >= 6 && cv::checkRange(rvec) && cv::checkRange(tvec);
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

bool ParseImageName(const std::string& image_name, bool& is_left, std::string& frame_id)
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

}  // namespace

// ─── UnionFind ───────────────────────────────────────────────────────────────

int UnionFind::AddNode()
{
  const int idx = static_cast<int>(parent_.size());
  parent_.push_back(idx);
  rank_.push_back(0);
  return idx;
}

int UnionFind::Find(int x)
{
  if (parent_[x] != x) {
    parent_[x] = Find(parent_[x]);
  }
  return parent_[x];
}

void UnionFind::Unite(int a, int b)
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

// ─── BuildTracks ────────────────────────────────────────────────────────────

bool BuildTracks(const std::vector<RawImagePair>& pairs,
                 double max_match_score,
                 int min_pair_inliers,
                 double min_pair_inlier_ratio,
                 int min_track_len,
                 TrackBuildResult& result)
{
  const double kFmRansacThreshold  = 2.0;
  const double kFmRansacConfidence = 0.995;
  const int    kMinMatchesForRansac = 8;

  std::vector<ImageInfo>& images = result.images;
  std::vector<FrameState>& frames = result.frames;
  std::vector<std::string>& frame_ids = result.frame_ids;
  std::vector<Track>& tracks = result.tracks;

  images.clear();
  frames.clear();
  frame_ids.clear();
  tracks.clear();

  std::unordered_map<std::string, int> image_index;

  auto ensure_image = [&](const std::string& name,
                          bool has_meta,
                          bool meta_is_left,
                          const std::string& meta_frame_id) {
    std::unordered_map<std::string, int>::const_iterator it = image_index.find(name);
    if (it != image_index.end()) {
      return it->second;
    }
    const int idx = static_cast<int>(images.size());
    image_index[name] = idx;

    ImageInfo info;
    info.name = name;
    if (has_meta && !meta_frame_id.empty()) {
      info.valid = true;
      info.is_left = meta_is_left;
      info.frame_id = meta_frame_id;
    } else {
      bool is_left = true;
      std::string frame_id;
      info.valid = ParseImageName(name, is_left, frame_id);
      info.is_left = is_left;
      info.frame_id = frame_id;
    }
    images.push_back(info);
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

  // ── Phase 1: Build union-find edges ───────────────────────────────────────
  for (size_t i = 0; i < pairs.size(); ++i) {
    const RawImagePair& pair = pairs[i];
    if (pair.image_a == pair.image_b) {
      continue;
    }

    const int ia = ensure_image(pair.image_a, pair.has_image_a_meta,
                                pair.image_a_is_left, pair.image_a_frame_id);
    const int ib = ensure_image(pair.image_b, pair.has_image_b_meta,
                                pair.image_b_is_left, pair.image_b_frame_id);

    std::vector<char> inlier_mask(pair.matches.size(), 1);

    std::vector<cv::Point2f> pts_a;
    std::vector<cv::Point2f> pts_b;
    std::vector<int> idx_map;
    pts_a.reserve(pair.matches.size());
    pts_b.reserve(pair.matches.size());
    idx_map.reserve(pair.matches.size());

    for (size_t k = 0; k < pair.matches.size(); ++k) {
      const RawPairMatch& m = pair.matches[k];
      if (m.score > max_match_score) {
        continue;
      }
      pts_a.push_back(m.pt_a);
      pts_b.push_back(m.pt_b);
      idx_map.push_back(static_cast<int>(k));
    }

    bool pair_accepted = (pts_a.size() >= static_cast<size_t>(kMinMatchesForRansac));
    if (pts_a.size() >= static_cast<size_t>(kMinMatchesForRansac)) {
      cv::Mat inliers;
      bool geometry_ok = false;

      {
        const cv::Mat F = cv::findFundamentalMat(pts_a, pts_b, cv::FM_RANSAC,
                                                 kFmRansacThreshold, kFmRansacConfidence, inliers);
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
        if (inlier_count < min_pair_inliers || inlier_ratio < min_pair_inlier_ratio) {
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
      if (m.score > max_match_score) {
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
  for (size_t i = 0; i < images.size(); ++i) {
    ImageInfo& img = images[i];
    if (!img.valid) {
      continue;
    }

    std::map<std::string, int>::const_iterator it = frame_map.find(img.frame_id);
    if (it == frame_map.end()) {
      const int fidx = static_cast<int>(frames.size());
      frame_map[img.frame_id] = fidx;
      FrameState frame;
      frame.frame_id = img.frame_id;
      frames.push_back(frame);
      frame_ids.push_back(img.frame_id);
      img.frame_idx = fidx;
    } else {
      img.frame_idx = it->second;
    }

    FrameState& f = frames[img.frame_idx];
    if (img.is_left) {
      f.left_image_idx = static_cast<int>(i);
    } else {
      f.right_image_idx = static_cast<int>(i);
    }
  }

  std::vector<FrameState> valid_frames;
  std::vector<std::string> valid_frame_ids;
  std::vector<int> old_to_new_frame(frames.size(), -1);

  for (size_t i = 0; i < frames.size(); ++i) {
    const FrameState& f = frames[i];
    if (f.left_image_idx < 0 || f.right_image_idx < 0) {
      continue;
    }
    old_to_new_frame[i] = static_cast<int>(valid_frames.size());
    valid_frames.push_back(f);
    valid_frame_ids.push_back(f.frame_id);
  }

  frames.swap(valid_frames);
  frame_ids.swap(valid_frame_ids);

  for (size_t i = 0; i < images.size(); ++i) {
    ImageInfo& img = images[i];
    if (img.frame_idx >= 0) {
      img.frame_idx = old_to_new_frame[img.frame_idx];
    }
  }

  if (frames.empty()) {
    std::cerr << "No valid stereo frame (left_xxx + right_xxx) found." << std::endl;
    return false;
  }

  std::unordered_map<int, std::vector<int> > comps;
  comps.reserve(nodes.size());

  // ── Phase 2: Resolve components and build tracks ───────────────────────────
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int root = uf.Find(static_cast<int>(i));
    comps[root].push_back(static_cast<int>(i));
  }

  result.num_observations = 0;

  for (std::unordered_map<int, std::vector<int> >::const_iterator it = comps.begin(); it != comps.end(); ++it) {
    const std::vector<int>& comp = it->second;
    Track track;
    std::set<std::pair<int, bool> > seen;

    for (size_t j = 0; j < comp.size(); ++j) {
      const NodeInfo& node = nodes[comp[j]];
      const ImageInfo& img = images[node.image_idx];
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

    if (track.observations.size() < static_cast<size_t>(min_track_len)) {
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
    tracks.push_back(track);
    result.num_observations += track.observations.size();
  }

  result.num_tracks = tracks.size();

  if (tracks.empty()) {
    std::cerr << "No valid tracks after union-find filtering." << std::endl;
    return false;
  }

  return true;
}

// ─── CollectLeftLeftCorrespondences ──────────────────────────────────────────

size_t CollectLeftLeftCorrespondences(const std::vector<Track>& tracks,
                                      int frame_a, int frame_b,
                                      std::vector<cv::Point2f>& pts_a,
                                      std::vector<cv::Point2f>& pts_b)
{
  pts_a.clear();
  pts_b.clear();

  for (size_t t = 0; t < tracks.size(); ++t) {
    const Track& track = tracks[t];
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

// ─── InitializeFrameRotations ───────────────────────────────────────────────

bool InitializeFrameRotations(const StereoCamera& init_camera,
                               const std::vector<RawImagePair>& pairs,
                               const std::vector<ImageInfo>& images,
                               double max_match_score,
                               int min_pair_inliers,
                               double min_pair_inlier_ratio,
                               const std::vector<Track>& tracks,
                               std::vector<FrameState>& frames,
                               std::vector<int>& registration_order,
                               int& fixed_frame_idx)
{
  if (frames.empty()) {
    return false;
  }
  registration_order.clear();

  const cv::Mat K = init_camera.left.K();
  const cv::Mat dist = init_camera.left.dist();

  int start_frame = 0;
  if (!SelectBootstrapStartFrame(pairs, images, max_match_score,
                                 min_pair_inliers, min_pair_inlier_ratio,
                                 start_frame)) {
    size_t best_count = 0;
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      size_t cnt = 0;
      for (size_t ti = 0; ti < tracks.size(); ++ti) {
        const Track& track = tracks[ti];
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
  }

  for (size_t i = 0; i < frames.size(); ++i) {
    frames[i].initialized = false;
    frames[i].rvec.assign(3, 0.0);
    frames[i].tvec.assign(3, 0.0);
  }

  frames[start_frame].initialized = true;
  fixed_frame_idx = start_frame;
  registration_order.push_back(start_frame);

  size_t registered = 1;
  std::vector<cv::Point2f> pts_from;
  std::vector<cv::Point2f> pts_to;

  while (registered < frames.size()) {
    const int best_from = registration_order.back();
    int best_to = -1;

    if (!SelectNextFrameFromPrevious(pairs, images, max_match_score,
                                     min_pair_inliers, min_pair_inlier_ratio,
                                     best_from, frames, best_to)) {
      for (size_t i = 0; i < frames.size(); ++i) {
        if (!frames[i].initialized) {
          frames[i].initialized = true;
          frames[i].rvec = frames[best_from].rvec;
          frames[i].tvec = frames[best_from].tvec;
          registration_order.push_back(static_cast<int>(i));
          registered++;
        }
      }
      break;
    }

    bool pose_initialized = false;
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    if (CollectStereoPnPData(init_camera, tracks, frames, best_from, best_to, object_points, image_points)) {
      cv::Mat rvec_to;
      cv::Mat tvec_to;
      int pnp_inliers = 0;
      if (EstimatePoseFromPnP(object_points, image_points, K, dist, frames[best_from],
                              rvec_to, tvec_to, pnp_inliers)) {
        frames[best_to].rvec = {
            rvec_to.at<double>(0, 0),
            rvec_to.at<double>(1, 0),
            rvec_to.at<double>(2, 0),
        };
        frames[best_to].tvec = {
            tvec_to.at<double>(0, 0),
            tvec_to.at<double>(1, 0),
            tvec_to.at<double>(2, 0),
        };
        pose_initialized = true;
      }
    }

    if (!pose_initialized) {
      const size_t overlap = CollectLeftLeftCorrespondences(tracks, best_from, best_to, pts_from, pts_to);
      if (overlap >= 8) {
        cv::Mat rel_rvec;
        cv::Mat rel_tvec;
        if (EstimatePoseFromEssential(pts_from, pts_to, K, dist, rel_rvec, rel_tvec)) {
          const cv::Mat R_from = camera_math::ToRotation(frames[best_from].rvec);
          cv::Mat R_rel;
          cv::Rodrigues(rel_rvec, R_rel);
          const cv::Mat R_to = R_rel * R_from;
          cv::Mat rvec_to;
          cv::Rodrigues(R_to, rvec_to);

          const cv::Mat t_from = ToColumnMat(frames[best_from].tvec);
          const cv::Mat t_to = t_from + rel_tvec;
          frames[best_to].rvec = {
              rvec_to.at<double>(0, 0),
              rvec_to.at<double>(1, 0),
              rvec_to.at<double>(2, 0),
          };
          frames[best_to].tvec = {
              t_to.at<double>(0, 0),
              t_to.at<double>(1, 0),
              t_to.at<double>(2, 0),
          };
          pose_initialized = true;
        }
      }
    }

    if (!pose_initialized) {
      frames[best_to].rvec = frames[best_from].rvec;
      frames[best_to].tvec = frames[best_from].tvec;
    }

    frames[best_to].initialized = true;
    registration_order.push_back(best_to);
    registered++;
  }

  return true;
}

// ─── InitializeTrackPoints ──────────────────────────────────────────────────

bool InitializeTrackPoints(const StereoCamera& init_camera,
                            const std::vector<double>& extrinsics,
                            const std::vector<FrameState>& frames,
                            std::vector<Track>& tracks)
{
  const cv::Mat K_l = init_camera.left.K();
  const cv::Mat dist_l = init_camera.left.dist();
  const cv::Mat K_r = init_camera.right.K();
  const cv::Mat dist_r = init_camera.right.dist();

  StereoExtrinsics ext = init_camera.extrinsics;
  if (ext.R.empty() || ext.t.empty()) {
    ext.FromVector(extrinsics);
  }

  cv::Mat P_l = cv::Mat::eye(3, 4, CV_64F);
  cv::Mat P_r(3, 4, CV_64F);
  ext.R.copyTo(P_r(cv::Range(0, 3), cv::Range(0, 3)));
  ext.t.copyTo(P_r(cv::Range(0, 3), cv::Range(3, 4)));

  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    Track& track = tracks[ti];
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

        const cv::Mat Xw = WorldFromCamera(frames[obs_l.frame_idx], Xl);

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

      const cv::Mat Xw = WorldFromCamera(frames[obs.frame_idx], 5.0 * ray_l);
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

}  // namespace stereocalib
