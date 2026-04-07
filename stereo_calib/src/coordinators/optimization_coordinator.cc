#include "coordinators/optimization_coordinator.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace stereocalib {

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
  bc.aspect_ratio_prior_weight = config.aspect_ratio_prior_weight;
  bc.baseline_prior_weight = config.baseline_prior_weight;
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

  // ── Step 2: Apply ground truth frame poses ────────────────────────────────
  ApplyFramePoses(frames);

  // ── Step 3: Frame rotation initialization ─────────────────────────────────
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
    std::cerr << "Frame rotation initialization failed." << std::endl;
    return result;
  }

  // ── Step 4: Initialize BA state ───────────────────────────────────────────
  BAState state;
  state.intrinsics_left = input.init_camera.left.ToVector();
  state.intrinsics_right = input.init_camera.right.ToVector();
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

  // ── Step 6: Incremental BA ────────────────────────────────────────────────
  const auto& reg_order = frame_init.registration_order;
  std::vector<char> active_frames(frames.size(), 0);
  active_frames[reg_order[0]] = 1;

  bool have_rmse = false;
  int successful_registrations = 0;

  for (size_t i = 1; i < reg_order.size(); ++i) {
    const int frame_idx = reg_order[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(active_frames.size())) {
      continue;
    }
    active_frames[frame_idx] = 1;
    successful_registrations++;

    // Incremental BA for new frame
    BAResult ba_result = ba_service_->RunBundleAdjustment(
        state, active_frames, ToBAConfig(config, config.incremental_max_iter), frame_idx);
    
    if (ba_result.success) {
      if (!have_rmse) {
        result.init_reproj_error = ba_result.init_rmse;
        have_rmse = true;
      }
      result.final_reproj_error = ba_result.final_rmse;
      
      std::cout << "[Incremental BA] registered_frames=" << (i + 1)
                << "/" << reg_order.size()
                << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                << ba_result.final_rmse << " px" << std::endl;

      std::string stage_name = "Incremental BA - Frame " + std::to_string(i + 1);
      StereoCamera current = BuildCamera(state);
      eval_service_->PrintCurrentVsGroundTruth(stage_name, current);
      eval_service_->RecordOptimizationStage(stage_name, ba_result.final_rmse, current);
    }

    // Periodic global BA
    if (config.global_opt_interval > 0 &&
        successful_registrations % config.global_opt_interval == 0) {
      BAResult global_result = ba_service_->RunBundleAdjustment(
          state, active_frames, ToBAConfig(config, config.max_iter));
      
      if (global_result.success) {
        if (!have_rmse) {
          result.init_reproj_error = global_result.init_rmse;
          have_rmse = true;
        }
        result.final_reproj_error = global_result.final_rmse;

        std::cout << "[Global BA] registered_frames=" << (i + 1)
                  << "/" << reg_order.size()
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << global_result.final_rmse << " px" << std::endl;

        std::string stage_name = "Periodic Global BA - Frame " + std::to_string(i + 1);
        StereoCamera current = BuildCamera(state);
        eval_service_->PrintCurrentVsGroundTruth(stage_name, current);
        eval_service_->RecordOptimizationStage(stage_name, global_result.final_rmse, current);
      }
    }
  }

  // ── Step 7: Final global BA ───────────────────────────────────────────────
  std::fill(active_frames.begin(), active_frames.end(), 1);
  BAResult final_ba = ba_service_->RunBundleAdjustment(
      state, active_frames, ToBAConfig(config, config.max_iter));
  
  if (!final_ba.success) {
    std::cerr << "Final global BA failed." << std::endl;
    return result;
  }

  result.init_reproj_error = final_ba.init_rmse;
  result.final_reproj_error = final_ba.final_rmse;

  std::cout << final_ba.summary.BriefReport() << std::endl;
  std::cout << "Tracks=" << result.num_tracks 
            << ", observations=" << result.num_observations 
            << ", frames=" << result.num_frames << std::endl;
  std::cout << "Reprojection error: init=" << std::fixed << std::setprecision(4) 
            << result.init_reproj_error
            << " px, final=" << result.final_reproj_error << " px" << std::endl;

  StereoCamera final_camera = BuildCamera(state);
  eval_service_->PrintCurrentVsGroundTruth("Final Global BA", final_camera);
  eval_service_->RecordOptimizationStage("Final Global BA", result.final_reproj_error, final_camera);

  // ── Step 8: Check convergence ─────────────────────────────────────────────
  const bool converged = (final_ba.summary.termination_type == ceres::CONVERGENCE ||
                          final_ba.summary.termination_type == ceres::NO_CONVERGENCE);
  const bool pass_reproj = (result.final_reproj_error <= config.max_reproj_error);

  if (!converged) {
    std::cerr << "Offline BA did not converge." << std::endl;
  }
  if (!pass_reproj) {
    std::cerr << "Final reprojection error " << result.final_reproj_error
              << " px exceeds threshold " << config.max_reproj_error << " px." << std::endl;
  }

  // ── Finalize result ───────────────────────────────────────────────────────
  result.camera = final_camera;
  result.optimization_history = eval_service_->GetOptimizationHistory();
  result.success = converged && pass_reproj;

  return result;
}

}  // namespace stereocalib
