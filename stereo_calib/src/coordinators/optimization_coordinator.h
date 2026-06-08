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
  int per_frame_max_iter = 5;
  int global_opt_interval = 5;
  double huber_delta = 1.0;
  bool fix_distortion = true;
  bool fix_focal_length = false;
  bool fix_principal_point = true;
  bool enable_per_frame_correction = true;
  double aspect_ratio_prior_weight = 100.0;
  double baseline_prior_weight = 10.0;
  double tx_prior_weight = 0.0;
  double focal_prior_weight = 0.0;
  double focal_mean_prior_weight = 0.0;
  double stereo_intrinsics_consistency_weight = 0.0;
  double principal_point_mean_prior_weight = 0.0;
  double frame_distance_prior_weight = 0.0;
  double frame_position_prior_weight = 0.0;
  double frame_translation_vector_prior_weight = 0.0;
  double frame_translation_direction_prior_weight = 0.0;
  int frame_distance_prior_stride = 1;
  int frame_distance_prior_max_stride = 1;
  double frame_rotation_angle_prior_weight = 0.0;
  double frame_rotation_vector_prior_weight = 0.0;
  double frame_absolute_rotation_prior_weight = 0.0;
  double focal_lower_scale = 0.5;
  double focal_upper_scale = 1.5;
  double per_frame_max_rmse = 5.0;
  double per_frame_max_rmse_growth = 3.0;
  bool normalize_initial_focal_to_mean = false;
  bool normalize_initial_stereo_translation_to_x_axis = false;
  bool initialize_frame_poses_from_external = false;
  bool fix_external_frame_poses = false;
  bool fix_external_frame_rotations = false;
  bool fix_external_frame_translations = false;
  bool reset_camera_params_each_ba_round = false;
  // Final global BA strategy. When true, final BA runs twice:
  // first with principal point fixed, then with principal point free.
  bool enable_two_stage_final_global_ba = false;
  bool enable_incremental_free_principal_point_refine = false;
  int incremental_free_principal_point_max_iter = 8;
  int min_active_frames_for_free_principal_point = 20;
  int free_principal_point_every_n_global_ba = 2;
  double free_principal_point_max_rmse_increase = 0.05;
  double free_principal_point_min_rmse_decrease = 0.005;
  // Max distance from the left/right initial principal-point mean after
  // temporarily freeing cx/cy.
  double free_principal_point_max_delta = 3.0;
  bool free_principal_point_fix_focal_length = true;
  double free_principal_point_max_focal_delta = 0.0;
  double final_free_principal_point_min_rmse_decrease = 0.02;
  double final_free_principal_point_max_delta = 3.0;
  bool final_free_principal_point_fix_focal_length = true;
  double final_free_principal_point_max_focal_delta = 0.0;
  bool optimize_stereo_tx_in_final_global_ba = false;
  bool enable_final_stereo_extrinsics_refine = false;
  int final_stereo_extrinsics_max_iter = 12;
  double final_stereo_extrinsics_min_rmse_decrease = 0.001;
  double final_stereo_extrinsics_max_translation_delta = 0.02;
  double final_stereo_extrinsics_max_rotation_delta = 0.002;
  double final_stereo_extrinsics_max_frame_distance_rms_increase = -1.0;
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
  size_t num_conflicted_components = 0;
  size_t num_conflict_observations_skipped = 0;
  size_t num_components_skipped_due_to_conflict = 0;
  
  double init_reproj_error = 0.0;
  double final_reproj_error = 0.0;
  
  std::vector<nlohmann::json> optimization_history;
  std::vector<nlohmann::json> outlier_rejection_history;
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

  /// Run the registration workflow with per-step active-set global BA.
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
  bool InitializeFramePosesFromExternal(std::vector<FrameState>& frames,
                                        std::vector<int>& registration_order,
                                        int& fixed_frame_idx) const;

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
