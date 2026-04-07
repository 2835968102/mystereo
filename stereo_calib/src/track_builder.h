/*
 * Track building, frame initialization and point initialization for
 * multi-frame offline stereo BA.
 */

#ifndef STEREO_CALIB_SRC_TRACK_BUILDER_H
#define STEREO_CALIB_SRC_TRACK_BUILDER_H

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

// ─── Types ───────────────────────────────────────────────────────────────────

struct TrackObservation {
  int frame_idx = -1;
  bool is_left = true;
  cv::Point2f px;
  bool rejected = false;
};

struct Track {
  std::vector<TrackObservation> observations;
  std::vector<double> point3d;  // world coordinates
};

struct FrameState {
  std::string frame_id;
  int left_image_idx = -1;
  int right_image_idx = -1;
  std::vector<double> rvec = {0.0, 0.0, 0.0};
  bool initialized = false;

  // Ground truth pose (if available)
  bool has_gt_pose = false;
  std::vector<double> gt_rvec = {0.0, 0.0, 0.0};  // Rodrigues rotation vector
  std::vector<double> gt_tvec = {0.0, 0.0, 0.0};  // Translation vector
};

struct ImageInfo {
  std::string name;
  bool valid = false;
  bool is_left = true;
  std::string frame_id;
  int frame_idx = -1;
};

// ─── Union-Find ──────────────────────────────────────────────────────────────

class UnionFind {
 public:
  int AddNode();
  int Find(int x);
  void Unite(int a, int b);

 private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

// ─── Track building result ───────────────────────────────────────────────────

struct TrackBuildResult {
  std::vector<Track> tracks;
  std::vector<FrameState> frames;
  std::vector<std::string> frame_ids;
  std::vector<ImageInfo> images;
  size_t num_tracks;
  size_t num_observations;
};

// ─── Core functions ──────────────────────────────────────────────────────────

bool BuildTracks(const std::vector<RawImagePair>& pairs,
                 double max_match_score,
                 int min_pair_inliers,
                 double min_pair_inlier_ratio,
                 int min_track_len,
                 TrackBuildResult& result);

bool InitializeFrameRotations(const StereoCamera& init_camera,
                               const std::vector<RawImagePair>& pairs,
                               const std::vector<ImageInfo>& images,
                               double max_match_score,
                               int min_pair_inliers,
                               double min_pair_inlier_ratio,
                               const std::vector<Track>& tracks,
                               std::vector<FrameState>& frames,
                               std::vector<int>& registration_order,
                               int& fixed_frame_idx);

bool InitializeTrackPoints(const StereoCamera& init_camera,
                            const std::vector<double>& extrinsics,
                            const std::vector<FrameState>& frames,
                            std::vector<Track>& tracks);

size_t CollectLeftLeftCorrespondences(const std::vector<Track>& tracks,
                                      int frame_a, int frame_b,
                                      std::vector<cv::Point2f>& pts_a,
                                      std::vector<cv::Point2f>& pts_b);

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_TRACK_BUILDER_H
