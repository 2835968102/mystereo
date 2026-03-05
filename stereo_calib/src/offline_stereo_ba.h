#ifndef STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H
#define STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

struct RawPairMatch {
  cv::Point2f pt_a;
  cv::Point2f pt_b;
  double score = 0.0;
};

struct RawImagePair {
  std::string image_a;
  std::string image_b;
  std::vector<RawPairMatch> matches;
};

struct OfflineBAInput {
  StereoCamera init_camera;
  std::vector<RawImagePair> pairs;
};

class OfflineStereoBA {
 public:
  struct Options {
    int max_iter = 200;
    int incremental_max_iter = 20;
    int global_opt_interval = 5;
    int min_track_len = 3;
    double huber_delta = 1.0;
    double max_match_score = 1.0;
    int min_pair_inliers = 12;
    double min_pair_inlier_ratio = 0.35;
    bool fix_distortion = false;
    double aspect_ratio_prior_weight = 1.0;
    double max_reproj_error = 20.0;
    double baseline_prior_weight = 10.0;
  };

  OfflineStereoBA(const OfflineBAInput& input, const Options& options);
  ~OfflineStereoBA() = default;

  bool Solve(StereoCamera& result);

  size_t num_tracks() const { return num_tracks_; }
  size_t num_observations() const { return num_observations_; }
  size_t num_frames() const { return frame_ids_.size(); }

  double init_reproj_error() const { return init_reproj_error_; }
  double final_reproj_error() const { return final_reproj_error_; }

 private:
  struct TrackObservation {
    int frame_idx = -1;
    bool is_left = true;
    cv::Point2f px;
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
  };

  struct ImageInfo {
    std::string name;
    bool valid = false;
    bool is_left = true;
    std::string frame_id;
    int frame_idx = -1;
  };

  class UnionFind {
   public:
    int AddNode();
    int Find(int x);
    void Unite(int a, int b);

   private:
    std::vector<int> parent_;
    std::vector<int> rank_;
  };

  bool BuildTracks();
  bool ParseImageName(const std::string& image_name, bool& is_left, std::string& frame_id) const;
  bool InitializeFrameRotations(std::vector<int>& registration_order);
  bool InitializeTrackPoints();
  size_t CollectLeftLeftCorrespondences(int frame_a, int frame_b,
                                        std::vector<cv::Point2f>& pts_a,
                                        std::vector<cv::Point2f>& pts_b) const;

  bool RunBundleAdjustment(const std::vector<char>& active_frames,
                           int max_num_iterations,
                           ceres::Solver::Summary& summary,
                           double& init_rmse,
                           double& final_rmse,
                           int frame_to_optimize = -1);
  void ApplyResult(StereoCamera& result);

 private:
  OfflineBAInput input_;
  Options options_;

  std::vector<ImageInfo> images_;
  std::vector<FrameState> frames_;
  std::vector<std::string> frame_ids_;
  std::vector<Track> tracks_;

  std::vector<double> intrinsics_left_;
  std::vector<double> intrinsics_right_;
  std::vector<double> extrinsics_;
  std::vector<double> init_extrinsics_;

  ceres::Solver::Summary summary_;

  size_t num_tracks_ = 0;
  size_t num_observations_ = 0;
  int fixed_frame_idx_ = 0;

  double init_reproj_error_ = 0.0;
  double final_reproj_error_ = 0.0;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H
