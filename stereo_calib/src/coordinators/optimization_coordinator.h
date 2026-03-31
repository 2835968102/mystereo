#ifndef STEREO_CALIB_SRC_COORDINATORS_OPTIMIZATION_COORDINATOR_H
#define STEREO_CALIB_SRC_COORDINATORS_OPTIMIZATION_COORDINATOR_H

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "stereo_types.h"
#include "track_builder.h"
#include "services/track_service.h"
#include "services/initialization_service.h"
#include "services/bundle_adjustment_service.h"
#include "services/outlier_rejection_service.h"
#include "services/evaluation_service.h"

namespace stereocalib {

// ─── Optimization Configuration ─────────────────────────────────────────────

struct OptimizationConfig {
  // Track building
  double max_match_score = 1.0;
  int min_pair_inliers = 12;
  double min_pair_inlier_ratio = 0.35;
  int min_track_len = 3;
  
  // BA optimization
  int max_iter = 200;
  int incremental_max_iter = 20;
  int global_opt_interval = 5;
  double huber_delta = 1.0;
  bool fix_distortion = true;
  double aspect_ratio_prior_weight = 100.0;
  double baseline_prior_weight = 10.0;
  double max_reproj_error = 20.0;
  
  // Outlier rejection
  double outlier_rejection_threshold = 2.0;
  int max_outlier_rejection_rounds = 100;
};

// ─── Optimization Result ────────────────────────────────────────────────────

struct OptimizationResult {
  bool success = false;
  StereoCamera camera;
  
  size_t num_tracks = 0;
  size_t num_observations = 0;
  size_t num_frames = 0;
  
  double init_reproj_error = 0.0;
  double final_reproj_error = 0.0;
  
  std::vector<nlohmann::json> optimization_history;
};

// ─── Optimization Coordinator ───────────────────────────────────────────────

class OptimizationCoordinator {
 public:
  /// Construct with dependency injection.
  OptimizationCoordinator(
      std::shared_ptr<ITrackService> track_service,
      std::shared_ptr<IInitializationService> init_service,
      std::shared_ptr<IBundleAdjustmentService> ba_service,
      std::shared_ptr<IOutlierRejectionService> outlier_service,
      std::shared_ptr<IEvaluationService> eval_service);
  
  /// Construct with default service implementations.
  OptimizationCoordinator();
  
  ~OptimizationCoordinator() = default;

  /// Run the full incremental BA workflow.
  /// @param input Input data (initial camera + image pairs).
  /// @param config Optimization configuration.
  /// @return Optimization result.
  OptimizationResult RunIncrementalBA(const OfflineBAInput& input,
                                      const OptimizationConfig& config);

  /// Set ground truth for evaluation.
  void SetGroundTruth(const StereoCamera& gt);
  
  /// Load frame poses from JSON.
  void LoadFramePoses(const nlohmann::json& poses_json);

 private:
  // Helper to convert config to service-specific configs
  TrackBuildConfig ToTrackConfig(const OptimizationConfig& config) const;
  BAConfig ToBAConfig(const OptimizationConfig& config, int max_iter) const;
  OutlierRejectionConfig ToOutlierConfig(const OptimizationConfig& config) const;
  
  // Build current camera from state
  StereoCamera BuildCamera(const BAState& state) const;
  
  // Apply frame poses from JSON to frames
  void ApplyFramePoses(std::vector<FrameState>& frames) const;

 private:
  std::shared_ptr<ITrackService> track_service_;
  std::shared_ptr<IInitializationService> init_service_;
  std::shared_ptr<IBundleAdjustmentService> ba_service_;
  std::shared_ptr<IOutlierRejectionService> outlier_service_;
  std::shared_ptr<IEvaluationService> eval_service_;
  
  nlohmann::json frame_poses_json_;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_COORDINATORS_OPTIMIZATION_COORDINATOR_H
