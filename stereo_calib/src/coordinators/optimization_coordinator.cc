#include "coordinators/optimization_coordinator.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace stereocalib {

namespace {

void ResetCameraParamsToInitialization(BAState& state) {
  state.intrinsics_left = state.init_intrinsics_left;
  state.intrinsics_right = state.init_intrinsics_right;
  state.extrinsics = state.init_extrinsics;
}

OutlierRejectionState BuildOutlierState(
    const BAState& state,
    std::vector<FrameState>& frames,
    std::vector<Track>& tracks,
    const std::vector<char>* active_frames) {
  OutlierRejectionState outlier_state;
  outlier_state.intrinsics_left = state.intrinsics_left;
  outlier_state.intrinsics_right = state.intrinsics_right;
  outlier_state.extrinsics = state.extrinsics;
  outlier_state.frames = &frames;
  outlier_state.tracks = &tracks;
  outlier_state.active_frames = active_frames;
  return outlier_state;
}

}  // namespace

// ─── Constructors ───────────────────────────────────────────────────────────

OptimizationCoordinator::OptimizationCoordinator(
    std::shared_ptr<ITrackService> track_service,
    std::shared_ptr<IInitializationService> init_service,
    std::shared_ptr<IBundleAdjustmentService> ba_service,
    std::shared_ptr<IOutlierRejectionService> outlier_service,
    std::shared_ptr<IEvaluationService> eval_service)
    : track_service_(std::move(track_service)),
      init_service_(std::move(init_service)),
      ba_service_(std::move(ba_service)),
      outlier_service_(std::move(outlier_service)),
      eval_service_(std::move(eval_service)) {}

OptimizationCoordinator::OptimizationCoordinator()
    : track_service_(std::make_shared<TrackService>()),
      init_service_(std::make_shared<InitializationService>()),
      ba_service_(std::make_shared<BundleAdjustmentService>()),
      outlier_service_(std::make_shared<OutlierRejectionService>()),
      eval_service_(std::make_shared<EvaluationService>()) {}

// ─── Configuration Conversion ───────────────────────────────────────────────

TrackBuildConfig OptimizationCoordinator::ToTrackConfig(
    const OptimizationConfig& config) const {
  TrackBuildConfig tc;
  tc.max_match_score = config.max_match_score;
  tc.min_pair_inliers = config.min_pair_inliers;
  tc.min_pair_inlier_ratio = config.min_pair_inlier_ratio;
  tc.min_track_len = config.min_track_len;
  return tc;
}

BAConfig OptimizationCoordinator::ToBAConfig(
    const OptimizationConfig& config, int max_iter) const {
  BAConfig bc;
  bc.max_iterations = max_iter;
  bc.huber_delta = config.huber_delta;
  bc.fix_distortion = config.fix_distortion;
  bc.fix_principal_point = config.fix_principal_point;
  bc.aspect_ratio_prior_weight = config.aspect_ratio_prior_weight;
  bc.baseline_prior_weight = config.baseline_prior_weight;
  bc.tx_prior_weight = config.tx_prior_weight;
  bc.focal_prior_weight = config.focal_prior_weight;
  bc.focal_lower_scale = config.focal_lower_scale;
  bc.focal_upper_scale = config.focal_upper_scale;
  return bc;
}

OutlierRejectionConfig OptimizationCoordinator::ToOutlierConfig(
    const OptimizationConfig& config) const {
  OutlierRejectionConfig oc;
  oc.threshold = config.outlier_rejection_threshold;
  oc.max_rounds = config.max_outlier_rejection_rounds;
  return oc;
}

StereoCamera OptimizationCoordinator::BuildCamera(const BAState& state) const {
  StereoCamera camera;
  camera.left.FromVector(state.intrinsics_left);
  camera.right.FromVector(state.intrinsics_right);
  camera.extrinsics.FromVector(state.extrinsics);
  return camera;
}

// ─── Ground Truth & Frame Poses ─────────────────────────────────────────────

void OptimizationCoordinator::SetGroundTruth(const StereoCamera& gt) {
  eval_service_->SetGroundTruth(gt);
}

void OptimizationCoordinator::LoadFramePoses(const nlohmann::json& poses_json) {
  frame_poses_json_ = poses_json;
}

void OptimizationCoordinator::ApplyFramePoses(
    std::vector<FrameState>& frames) const {
  if (frame_poses_json_.empty() || !frame_poses_json_.contains("frames")) {
    return;
  }

  for (auto& frame : frames) {
    for (const auto& pose : frame_poses_json_["frames"]) {
      if (pose.contains("frame_id") && pose["frame_id"] == frame.frame_id) {
        if (pose.contains("rotation")) {
          const auto& rot = pose["rotation"];
          frame.gt_rvec = {rot[0].get<double>(), rot[1].get<double>(), rot[2].get<double>()};
          frame.has_gt_pose = true;
        }
        break;
      }
    }
  }
}

// ─── Main Workflow ──────────────────────────────────────────────────────────

OptimizationResult OptimizationCoordinator::RunIncrementalBA(
    const OfflineBAInput& input,
    const OptimizationConfig& config) {
  
  OptimizationResult result;
  eval_service_->ClearHistory();

  // ── Step 1: Track building ────────────────────────────────────────────────
  TrackBuildResult build_result;
  if (!track_service_->BuildTracks(input.pairs, ToTrackConfig(config), build_result)) {
    std::cerr << "Track building failed." << std::endl;
    return result;
  }

  std::vector<Track> tracks = std::move(build_result.tracks);
  std::vector<FrameState> frames = std::move(build_result.frames);
  const std::vector<ImageInfo>& images = build_result.images;
  result.num_tracks = build_result.num_tracks;
  result.num_observations = build_result.num_observations;
  result.num_frames = frames.size();
  result.num_conflicted_components = build_result.num_conflicted_components;
  result.num_conflict_observations_skipped = build_result.num_conflict_observations_skipped;
  result.num_components_skipped_due_to_conflict = build_result.num_components_skipped_due_to_conflict;

  std::cout << "[Track Build] tracks=" << result.num_tracks
            << ", observations=" << result.num_observations
            << ", conflicted_components=" << result.num_conflicted_components
            << ", conflict_obs_skipped=" << result.num_conflict_observations_skipped
            << ", components_skipped=" << result.num_components_skipped_due_to_conflict
            << std::endl;

  // ── Step 2: Apply ground truth frame poses ────────────────────────────────
  ApplyFramePoses(frames);

  // ── Step 3: Frame pose initialization ─────────────────────────────────────
  FrameInitResult frame_init = init_service_->InitializeFrameRotations(
      input.init_camera,
      input.pairs,
      images,
      config.max_match_score,
      config.min_pair_inliers,
      config.min_pair_inlier_ratio,
      tracks,
      frames);
  if (!frame_init.success || frame_init.registration_order.empty()) {
    std::cerr << "Frame pose initialization failed." << std::endl;
    return result;
  }

  // ── Step 4: Initialize BA state ───────────────────────────────────────────
  BAState state;
  state.intrinsics_left = input.init_camera.left.ToVector();
  state.intrinsics_right = input.init_camera.right.ToVector();
  state.init_intrinsics_left = state.intrinsics_left;
  state.init_intrinsics_right = state.intrinsics_right;
  state.extrinsics = input.init_camera.extrinsics.ToVector();
  state.init_extrinsics = state.extrinsics;
  state.frames = &frames;
  state.tracks = &tracks;
  state.fixed_frame_idx = frame_init.fixed_frame_idx;

  // ── Step 5: Track point initialization ────────────────────────────────────
  PointInitResult point_init = init_service_->InitializeTrackPoints(
      input.init_camera, state.extrinsics, frames, tracks);
  if (!point_init.success) {
    std::cerr << "Track point initialization failed." << std::endl;
    return result;
  }

  // ── Step 6: Active-set global BA on interval ─────────────────────────────
  const auto& reg_order = frame_init.registration_order;
  std::vector<char> active_frames(frames.size(), 0);
  active_frames[reg_order[0]] = 1;

  const int global_opt_interval = std::max(1, config.global_opt_interval);
  BAConfig local_ba_config = ToBAConfig(config, config.per_frame_max_iter);
  local_ba_config.fix_camera_params = true;
  local_ba_config.fix_track_points = false;
  BAConfig incremental_ba_config = ToBAConfig(config, config.incremental_max_iter);
  if (config.enable_two_stage_final_global_ba) {
    incremental_ba_config.fix_principal_point = true;
  }
  const OutlierRejectionConfig outlier_config = ToOutlierConfig(config);

  bool have_rmse = false;
  int successful_registrations = 0;

  for (size_t i = 1; i < reg_order.size(); ++i) {
    const int frame_idx = reg_order[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(active_frames.size())) {
      continue;
    }
    active_frames[frame_idx] = 1;
    successful_registrations++;

    if (config.enable_per_frame_correction) {
      BAResult local_ba_result = ba_service_->RunBundleAdjustment(
          state, active_frames, local_ba_config, frame_idx);
      if (local_ba_result.success) {
        result.final_reproj_error = local_ba_result.final_rmse;
        if (!have_rmse) {
          result.init_reproj_error = local_ba_result.init_rmse;
          have_rmse = true;
        }
        std::cout << "[Per-Frame Correction] registered_frames=" << (i + 1)
                  << "/" << reg_order.size()
                  << ", target_frame=" << frame_idx
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << local_ba_result.final_rmse << " px" << std::endl;

        StereoCamera current = BuildCamera(state);
        eval_service_->RecordOptimizationStage(
            "Per-Frame Correction - Registered Frame " + std::to_string(i + 1),
            local_ba_result.final_rmse,
            current);
      }
    }

    const bool is_interval_step = (successful_registrations % global_opt_interval) == 0;
    const bool is_last_registration = (i + 1 == reg_order.size());
    if (!is_interval_step && !is_last_registration) {
      continue;
    }

    if (config.reset_camera_params_each_ba_round) {
      ResetCameraParamsToInitialization(state);
    }

    BAResult ba_result = ba_service_->RunBundleAdjustment(
        state, active_frames, incremental_ba_config, -1);
    if (!ba_result.success) {
      continue;
    }

    if (!have_rmse) {
      result.init_reproj_error = ba_result.init_rmse;
      have_rmse = true;
    }

    OutlierRejectionState interval_outlier_state =
        BuildOutlierState(state, frames, tracks, &active_frames);
    const OutlierRejectionResult rejection_result =
        outlier_service_->RejectOutliersIterative(interval_outlier_state, outlier_config);
    std::cout << "[Interval Outlier Rejection] registered_frames=" << (i + 1)
              << "/" << reg_order.size()
              << ", rejected=" << rejection_result.rejected_count
              << ", rounds=" << rejection_result.total_rounds
              << ", threshold=" << std::fixed << std::setprecision(4)
              << outlier_config.threshold << " px" << std::endl;

    if (rejection_result.rejected_count > 0) {
      if (config.reset_camera_params_each_ba_round) {
        ResetCameraParamsToInitialization(state);
      }
      BAResult refined_ba_result = ba_service_->RunBundleAdjustment(
          state, active_frames, incremental_ba_config, -1);
      if (refined_ba_result.success) {
        ba_result = refined_ba_result;
      }
    }

    result.final_reproj_error = ba_result.final_rmse;

    std::cout << "[Interval Active-Set Global BA] registered_frames=" << (i + 1)
              << "/" << reg_order.size()
              << ", reproj_rmse=" << std::fixed << std::setprecision(4)
              << ba_result.final_rmse << " px" << std::endl;

    std::string stage_name = "Interval Global BA - Registered Frame " + std::to_string(i + 1);
    StereoCamera current = BuildCamera(state);
    eval_service_->PrintCurrentVsGroundTruth(stage_name, current);
    eval_service_->RecordOptimizationStage(stage_name, ba_result.final_rmse, current);
  }

  // ── Step 7: Post-pass outlier rejection + final global BA ────────────────
  if (!have_rmse) {
    std::cerr << "No interval BA result available." << std::endl;
    return result;
  }

  if (config.reset_camera_params_each_ba_round) {
    ResetCameraParamsToInitialization(state);
  }

  OutlierRejectionState outlier_state =
      BuildOutlierState(state, frames, tracks, &active_frames);
  const OutlierRejectionResult final_rejection_result =
      outlier_service_->RejectOutliersIterative(outlier_state, outlier_config);
  std::cout << "[Post BA Outlier Rejection] rejected="
            << final_rejection_result.rejected_count
            << ", rounds=" << final_rejection_result.total_rounds
            << ", threshold=" << std::fixed << std::setprecision(4)
            << outlier_config.threshold << " px" << std::endl;

  BAConfig final_ba_config = ToBAConfig(config, config.max_iter);
  BAResult final_ba_result;
  if (config.enable_two_stage_final_global_ba) {
    BAConfig fixed_principal_point_config = final_ba_config;
    fixed_principal_point_config.fix_principal_point = true;
    BAResult fixed_principal_point_result = ba_service_->RunBundleAdjustment(
        state, active_frames, fixed_principal_point_config, -1);
    if (fixed_principal_point_result.success) {
      result.final_reproj_error = fixed_principal_point_result.final_rmse;
      std::cout << "[Final Global BA (Fixed Principal Point)] registered_frames="
                << (successful_registrations + 1)
                << "/" << reg_order.size()
                << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                << fixed_principal_point_result.final_rmse << " px" << std::endl;

      StereoCamera current = BuildCamera(state);
      eval_service_->PrintCurrentVsGroundTruth("Final Global BA (Fixed Principal Point)", current);
      eval_service_->RecordOptimizationStage("Final Global BA (Fixed Principal Point)",
                                             fixed_principal_point_result.final_rmse,
                                             current);

      BAConfig free_principal_point_config = final_ba_config;
      free_principal_point_config.fix_principal_point = false;
      final_ba_result = ba_service_->RunBundleAdjustment(
          state, active_frames, free_principal_point_config, -1);
      if (final_ba_result.success) {
        result.final_reproj_error = final_ba_result.final_rmse;
        std::cout << "[Final Global BA (Free Principal Point)] registered_frames="
                  << (successful_registrations + 1)
                  << "/" << reg_order.size()
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << final_ba_result.final_rmse << " px" << std::endl;

        current = BuildCamera(state);
        eval_service_->PrintCurrentVsGroundTruth("Final Global BA (Free Principal Point)", current);
        eval_service_->RecordOptimizationStage("Final Global BA (Free Principal Point)",
                                               final_ba_result.final_rmse,
                                               current);
      } else {
        std::cerr << "Final global BA with free principal point failed after fixed-principal-point refinement." << std::endl;
        final_ba_result = fixed_principal_point_result;
      }
    } else {
      std::cerr << "Final global BA with fixed principal point failed after outlier rejection." << std::endl;
      final_ba_result = fixed_principal_point_result;
    }
  } else {
    final_ba_result = ba_service_->RunBundleAdjustment(
        state, active_frames, final_ba_config, -1);
    if (final_ba_result.success) {
      result.final_reproj_error = final_ba_result.final_rmse;
      std::cout << "[Final Global BA] registered_frames=" << (successful_registrations + 1)
                << "/" << reg_order.size()
                << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                << final_ba_result.final_rmse << " px" << std::endl;

      StereoCamera current = BuildCamera(state);
      eval_service_->PrintCurrentVsGroundTruth("Final Global BA", current);
      eval_service_->RecordOptimizationStage("Final Global BA",
                                             final_ba_result.final_rmse,
                                             current);
    } else {
      std::cerr << "Final global BA failed after outlier rejection." << std::endl;
    }
  }

  // ── Step 8: Finalization ───────────────────────────────────────────────────
  StereoCamera final_camera = BuildCamera(state);

  std::cout << "Tracks=" << result.num_tracks
            << ", observations=" << result.num_observations
            << ", frames=" << result.num_frames << std::endl;
  std::cout << "Reprojection error: final=" << std::fixed << std::setprecision(4)
            << result.final_reproj_error << " px" << std::endl;

  const bool pass_reproj = final_ba_result.success &&
                           (result.final_reproj_error <= config.max_reproj_error);
  if (!pass_reproj) {
    std::cerr << "Final reprojection error " << result.final_reproj_error
              << " px exceeds threshold " << config.max_reproj_error << " px."
              << std::endl;
  }

  // ── Finalize result ───────────────────────────────────────────────────────
  result.camera = final_camera;
  result.optimization_history = eval_service_->GetOptimizationHistory();
  result.success = pass_reproj;

  return result;
}

}  // namespace stereocalib
