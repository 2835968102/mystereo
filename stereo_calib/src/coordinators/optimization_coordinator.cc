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

struct BAStateSnapshot {
  std::vector<double> intrinsics_left;
  std::vector<double> intrinsics_right;
  std::vector<double> extrinsics;
  std::vector<std::vector<double>> frame_rvecs;
  std::vector<std::vector<double>> frame_tvecs;
  std::vector<std::vector<double>> track_points;
};

BAStateSnapshot SaveBAStateSnapshot(
    const BAState& state,
    const std::vector<FrameState>& frames,
    const std::vector<Track>& tracks) {
  BAStateSnapshot snapshot;
  snapshot.intrinsics_left = state.intrinsics_left;
  snapshot.intrinsics_right = state.intrinsics_right;
  snapshot.extrinsics = state.extrinsics;
  snapshot.frame_rvecs.reserve(frames.size());
  snapshot.frame_tvecs.reserve(frames.size());
  for (const FrameState& frame : frames) {
    snapshot.frame_rvecs.push_back(frame.rvec);
    snapshot.frame_tvecs.push_back(frame.tvec);
  }
  snapshot.track_points.reserve(tracks.size());
  for (const Track& track : tracks) {
    snapshot.track_points.push_back(track.point3d);
  }
  return snapshot;
}

void RestoreBAStateSnapshot(
    const BAStateSnapshot& snapshot,
    BAState& state,
    std::vector<FrameState>& frames,
    std::vector<Track>& tracks) {
  state.intrinsics_left = snapshot.intrinsics_left;
  state.intrinsics_right = snapshot.intrinsics_right;
  state.extrinsics = snapshot.extrinsics;
  for (size_t fi = 0; fi < frames.size() && fi < snapshot.frame_rvecs.size(); ++fi) {
    frames[fi].rvec = snapshot.frame_rvecs[fi];
    frames[fi].tvec = snapshot.frame_tvecs[fi];
  }
  for (size_t ti = 0; ti < tracks.size() && ti < snapshot.track_points.size(); ++ti) {
    tracks[ti].point3d = snapshot.track_points[ti];
  }
}

nlohmann::json BuildOutlierHistoryEntry(
    const std::string& stage,
    size_t registered_frames,
    size_t total_frames,
    double threshold,
    const OutlierRejectionResult& result) {
  nlohmann::json rounds = nlohmann::json::array();
  for (const OutlierRejectionRoundInfo& round : result.rounds) {
    rounds.push_back({
        {"round", round.round},
        {"observations_before", round.observations_before},
        {"rejected_count", round.rejected_count},
        {"remaining_observations", round.remaining_observations},
    });
  }

  return {
      {"stage", stage},
      {"registered_frames", registered_frames},
      {"total_frames", total_frames},
      {"threshold_px", threshold},
      {"total_rounds", result.total_rounds},
      {"rejected_count", result.rejected_count},
      {"initial_observations", result.initial_observations},
      {"remaining_observations", result.remaining_observations},
      {"rounds", rounds},
  };
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

  // KITTI RAW 序列的 BA 调度中心。
  // 这里不直接写 Ceres 残差，而是决定“哪些帧/点参与本轮优化”和
  // “何时做局部校正、全局 BA、外点剔除”。真正的 Ceres Problem
  // 在 BundleAdjustmentService::RunBundleAdjustment 中构建。

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
  // 根据双目匹配和初始相机参数估计每帧位姿，并给出注册顺序。
  // fixed_frame_idx 作为世界坐标参考帧，后续 BA 中会固定它的位姿，
  // 避免整个重建发生 gauge freedom 漂移。
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
  // BAState 是所有优化变量的可变容器：
  // 左/右内参、双目外参、每帧位姿、每条 track 的 3D 点都会被 Ceres
  // 直接写回这些 vector/struct 中。init_* 保留初值，用于 prior 和可选重置。
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
  // 在第一次 BA 前先三角化/初始化 track 的 3D 点。没有合理 3D 初值时，
  // 后面的重投影残差很容易落到坏的局部极值。
  PointInitResult point_init = init_service_->InitializeTrackPoints(
      input.init_camera, state.extrinsics, frames, tracks);
  if (!point_init.success) {
    std::cerr << "Track point initialization failed." << std::endl;
    return result;
  }

  // ── Step 6: Active-set global BA on interval ─────────────────────────────
  // active_frames 表示当前已经注册进来的帧。KITTI 序列不是一次性把所有
  // 帧丢进 BA，而是按 registration_order 逐步扩张 active set：
  //   - 可选 per-frame correction：只放开新帧位姿，固定双目相机参数；
  //   - interval global BA：每注册 global_opt_interval 帧，优化所有 active 帧、
  //     双目内外参和 3D 点；
  //   - 最后一帧也会强制触发一次 interval global BA。
  // 因此这里的“全局”指当前 active set 内的联合优化，而不是简单优化
  // 单帧或单个 stereo pair；随着 active set 扩大，优化问题逐步接近全序列 BA。
  const auto& reg_order = frame_init.registration_order;
  std::vector<char> active_frames(frames.size(), 0);
  active_frames[reg_order[0]] = 1;

  const int global_opt_interval = std::max(1, config.global_opt_interval);
  BAConfig local_ba_config = ToBAConfig(config, config.per_frame_max_iter);
  // 局部校正只用于稳定刚加入的帧：固定左右相机内参和双目外参，
  // 只优化目标帧位姿以及被观测到的 3D 点。
  local_ba_config.fix_camera_params = true;
  local_ba_config.fix_track_points = false;
  BAConfig incremental_ba_config = ToBAConfig(config, config.incremental_max_iter);
  if (config.enable_two_stage_final_global_ba) {
    // KITTI 默认最终会先固定主点、再释放主点。增量阶段也固定主点，
    // 可以减少早期 active set 较小时 cx/cy 与位姿互相补偿造成的不稳定。
    incremental_ba_config.fix_principal_point = true;
  }
  const OutlierRejectionConfig outlier_config = ToOutlierConfig(config);

  bool have_rmse = false;
  int successful_registrations = 0;
  int interval_global_ba_count = 0;

  for (size_t i = 1; i < reg_order.size(); ++i) {
    const int frame_idx = reg_order[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(active_frames.size())) {
      continue;
    }
    active_frames[frame_idx] = 1;
    successful_registrations++;

    if (config.enable_per_frame_correction) {
      // 可选本地 BA：只调整当前新注册帧，快速把它拉到已有 active set 上。
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
      // 实验开关：每轮全局 BA 前把相机参数拉回初值，只让当前 active set
      // 重新解释相机参数，便于比较“累积优化”和“每轮从初值开始”的差异。
      ResetCameraParamsToInitialization(state);
    }

    // 周期性 active-set global BA：frame_to_optimize=-1 表示不限定单帧，
    // 由 BundleAdjustmentService 根据 active_frames 放开所有 active 帧。
    // 这一轮会联合优化：
    //   1. 左右目内参（受 fix_distortion / fix_principal_point 控制）；
    //   2. 双目外参 extrinsics；
    //   3. active 帧的 rvec/tvec（参考帧 fixed_frame_idx 仍固定）；
    //   4. 被 active 观测约束到的 track 3D 点。
    BAResult ba_result = ba_service_->RunBundleAdjustment(
        state, active_frames, incremental_ba_config, -1);
    if (!ba_result.success) {
      continue;
    }
    interval_global_ba_count++;

    if (!have_rmse) {
      result.init_reproj_error = ba_result.init_rmse;
      have_rmse = true;
    }

    OutlierRejectionState interval_outlier_state =
        BuildOutlierState(state, frames, tracks, &active_frames);
    // 用刚优化出的相机/位姿/3D 点计算重投影误差，剔除当前 active set
    // 中超过阈值的观测；如果确实剔除了点，会立即再跑一次 BA 细化。
    const OutlierRejectionResult rejection_result =
        outlier_service_->RejectOutliersIterative(interval_outlier_state, outlier_config);
    result.outlier_rejection_history.push_back(BuildOutlierHistoryEntry(
        "Interval Outlier Rejection - Registered Frame " + std::to_string(i + 1),
        i + 1,
        reg_order.size(),
        outlier_config.threshold,
        rejection_result));
    std::cout << "[Interval Outlier Rejection] registered_frames=" << (i + 1)
              << "/" << reg_order.size()
              << ", rejected=" << rejection_result.rejected_count
              << ", rounds=" << rejection_result.total_rounds
              << ", remaining_observations=" << rejection_result.remaining_observations
              << ", threshold=" << std::fixed << std::setprecision(4)
              << outlier_config.threshold << " px" << std::endl;

    if (rejection_result.rejected_count > 0) {
      if (config.reset_camera_params_each_ba_round) {
        ResetCameraParamsToInitialization(state);
      }
      // 外点剔除改变了参与目标函数的观测集合，立刻再跑一次同样的
      // active-set global BA，让相机、位姿和 3D 点在干净观测上重新收敛。
      BAResult refined_ba_result = ba_service_->RunBundleAdjustment(
          state, active_frames, incremental_ba_config, -1);
      if (refined_ba_result.success) {
        ba_result = refined_ba_result;
      }
    }

    const int active_frame_count =
        static_cast<int>(std::count(active_frames.begin(), active_frames.end(), 1));
    const int free_refine_interval = std::max(1, config.free_principal_point_every_n_global_ba);
    const double max_free_refine_rmse_increase =
        std::max(0.0, config.free_principal_point_max_rmse_increase);
    const bool has_enough_active_frames =
        active_frame_count >= config.min_active_frames_for_free_principal_point;
    const bool is_scheduled_free_refine =
        (interval_global_ba_count % free_refine_interval) == 0;
    if (config.enable_incremental_free_principal_point_refine &&
        has_enough_active_frames &&
        is_scheduled_free_refine) {
      BAStateSnapshot fixed_principal_point_snapshot =
          SaveBAStateSnapshot(state, frames, tracks);
      const double fixed_principal_point_rmse = ba_result.final_rmse;

      BAConfig free_principal_point_config =
          ToBAConfig(config, config.incremental_free_principal_point_max_iter);
      free_principal_point_config.fix_principal_point = false;
      BAResult free_principal_point_result = ba_service_->RunBundleAdjustment(
          state, active_frames, free_principal_point_config, -1);

      if (free_principal_point_result.success &&
          free_principal_point_result.final_rmse <=
              fixed_principal_point_rmse + max_free_refine_rmse_increase) {
        ba_result = free_principal_point_result;
        std::cout << "[Interval Free Principal Point Refine] registered_frames="
                  << active_frame_count << "/" << reg_order.size()
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << free_principal_point_result.final_rmse << " px"
                  << ", fixed_principal_point_rmse=" << fixed_principal_point_rmse
                  << " px" << std::endl;

        StereoCamera current = BuildCamera(state);
        eval_service_->PrintCurrentVsGroundTruth(
            "Interval Free Principal Point Refine - Registered Frame " + std::to_string(i + 1),
            current);
        eval_service_->RecordOptimizationStage(
            "Interval Free Principal Point Refine - Registered Frame " + std::to_string(i + 1),
            free_principal_point_result.final_rmse,
            current);
      } else {
        RestoreBAStateSnapshot(fixed_principal_point_snapshot, state, frames, tracks);
        const double attempted_rmse = free_principal_point_result.success
                                          ? free_principal_point_result.final_rmse
                                          : -1.0;
        std::cout << "[Interval Free Principal Point Refine] rejected, registered_frames="
                  << active_frame_count << "/" << reg_order.size()
                  << ", fixed_principal_point_rmse=" << std::fixed << std::setprecision(4)
                  << fixed_principal_point_rmse << " px"
                  << ", attempted_rmse=" << attempted_rmse << " px" << std::endl;
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
  // 所有可注册帧加入后，再基于全量 active set 做一次外点清理和最终 BA。
  // KITTI 入口默认开启两阶段最终 BA：先固定主点求稳，再释放主点微调。
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
  result.outlier_rejection_history.push_back(BuildOutlierHistoryEntry(
      "Post BA Outlier Rejection",
      successful_registrations + 1,
      reg_order.size(),
      outlier_config.threshold,
      final_rejection_result));
  std::cout << "[Post BA Outlier Rejection] rejected="
            << final_rejection_result.rejected_count
            << ", rounds=" << final_rejection_result.total_rounds
            << ", remaining_observations=" << final_rejection_result.remaining_observations
            << ", threshold=" << std::fixed << std::setprecision(4)
            << outlier_config.threshold << " px" << std::endl;

  BAConfig final_ba_config = ToBAConfig(config, config.max_iter);
  BAResult final_ba_result;
  if (config.enable_two_stage_final_global_ba) {
    // Two-stage final BA 的核心就在这个分支：
    // 两次调用同一个 RunBundleAdjustment，区别只在 fix_principal_point。
    // 第一阶段先冻结主点，减少 cx/cy 和位姿/焦距之间的耦合；第二阶段
    // 以上一阶段结果为初值释放主点，做最终微调。
    BAConfig fixed_principal_point_config = final_ba_config;
    fixed_principal_point_config.fix_principal_point = true;
    // 第一阶段固定 cx/cy，优化焦距、畸变（如未固定）、双目外参、帧位姿和点。
    // 这是一次全量 active-set global BA，迭代次数使用 max_iter，而不是
    // incremental_max_iter；目的是在外点清理后充分收敛。
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
      // 第二阶段释放 cx/cy，在第一阶段结果附近做最终细调。
      // 如果这一阶段失败，会回退到固定主点阶段的结果，避免丢掉可用解。
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
    // 非两阶段模式：直接用最终配置做一次全量 active-set global BA。
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
