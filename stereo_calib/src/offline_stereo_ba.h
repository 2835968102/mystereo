#ifndef STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H
#define STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H

#include <ceres/ceres.h>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "stereo_types.h"
#include "track_builder.h"

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
    bool fix_distortion = true;
    double aspect_ratio_prior_weight = 100.0;
    double max_reproj_error = 20.0;
    double baseline_prior_weight = 10.0;
    // Post-BA iterative outlier rejection
    double outlier_rejection_threshold = 2.0;  // pixels
    int    max_outlier_rejection_rounds = 100;
  };

  OfflineStereoBA(const OfflineBAInput& input, const Options& options);
  ~OfflineStereoBA() = default;

  bool Solve(StereoCamera& result);

  void SetGroundTruth(const StereoCamera& gt);

  size_t num_tracks() const { return num_tracks_; }
  size_t num_observations() const { return num_observations_; }
  size_t num_frames() const { return frame_ids_.size(); }

  double init_reproj_error() const { return init_reproj_error_; }
  double final_reproj_error() const { return final_reproj_error_; }

  std::vector<nlohmann::json> GetOptimizationHistory() const { return optimization_history_; }

 private:
  bool RunBundleAdjustment(const std::vector<char>& active_frames,
                           int max_num_iterations,
                           ceres::Solver::Summary& summary,
                           double& init_rmse,
                           double& final_rmse,
                           int frame_to_optimize = -1);
  int  RejectOutliers(double threshold);
  void ApplyResult(StereoCamera& result);

  void PrintCurrentVsGroundTruth(const std::string& stage_name) const;
  void RecordOptimizationStage(const std::string& stage_name, double reproj_error);

 private:
  OfflineBAInput input_;
  Options options_;

  // Track-builder results (owned here after building)
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

  bool has_ground_truth_ = false;
  StereoCamera ground_truth_;

  std::vector<nlohmann::json> optimization_history_;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_OFFLINE_STEREO_BA_H
