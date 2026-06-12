#include "services/bundle_adjustment_service.h"

#include <cmath>

#include "core/ceres_compat.h"
#include "stereo_factors.h"

namespace stereocalib {

BAResult BundleAdjustmentService::RunBundleAdjustment(
    BAState& state,
    const std::vector<char>& active_frames,
    const BAConfig& config,
    int frame_to_optimize) {
  
  BAResult result;
  
  if (!state.frames || !state.tracks) {
    return result;
  }
  
  std::vector<FrameState>& frames = *state.frames;
  std::vector<Track>& tracks = *state.tracks;
  
  ceres::Problem problem;
  size_t active_residuals = 0;

  // Ceres 优化目标：
  //   min sum rho(||project(track point, frame pose, stereo camera) - observed pixel||^2)
  // 参数块包括左右目内参 [9]、双目外参 [6]、每帧位姿 r/t [3+3]、
  // 以及每条 track 的 3D 点 [3]。active_frames 控制本轮使用哪些帧；
  // frame_to_optimize >= 0 时只放开该帧位姿，用于 per-frame correction。
  // 全局 BA 的调用方式是 frame_to_optimize = -1：所有 active 帧的观测
  // 都会进入同一个 Ceres Problem，并与共享的相机内外参、共享的 3D tracks
  // 一起优化。

  // Add track reprojection residuals.
  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    const Track& track = tracks[ti];

    // 同一条 track 至少要有两个 active 观测，才对相机/位姿/点有约束价值。
    // 对全局 BA 来说，这一步会把跨帧 track 变成连接多个帧位姿的约束边。
    int active_obs = 0;
    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames.size())) {
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

    // 每个观测贡献 2 维重投影残差；Huber loss 降低少量坏匹配的影响。
    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) || !active_frames[obs.frame_idx]) {
        continue;
      }

      ceres::CostFunction* cost = TrackReprojFactor::Create(obs.px, obs.is_left);
      ceres::LossFunction* loss = new ceres::HuberLoss(config.huber_delta);
      problem.AddResidualBlock(cost,
                               loss,
                               state.intrinsics_left.data(),
                               state.intrinsics_right.data(),
                               state.extrinsics.data(),
                               frames[obs.frame_idx].rvec.data(),
                               frames[obs.frame_idx].tvec.data(),
                               tracks[ti].point3d.data());
      active_residuals++;
    }
  }

  if (active_residuals == 0) {
    return result;
  }

  // Add priors.
  // KITTI 的 rectified stereo 初值通常比较可靠，这些软约束用于防止
  // 双目 baseline、tx、焦距或 fx/fy 比例在弱约束场景中过度漂移。
  if (config.baseline_prior_weight > 0.0) {
    ceres::CostFunction* prior_cost = BaselinePriorFactor::Create(
        state.init_extrinsics, config.baseline_prior_weight);
    problem.AddResidualBlock(prior_cost, nullptr, state.extrinsics.data());
  }

  if (config.tx_prior_weight > 0.0) {
    ceres::CostFunction* tx_prior_cost = TxPriorFactor::Create(
        state.init_extrinsics, config.tx_prior_weight);
    problem.AddResidualBlock(tx_prior_cost, nullptr, state.extrinsics.data());
  }

  if (config.aspect_ratio_prior_weight > 0.0) {
    ceres::CostFunction* aspect_left = AspectRatioPriorFactor::Create(
        config.aspect_ratio_prior_weight);
    ceres::CostFunction* aspect_right = AspectRatioPriorFactor::Create(
        config.aspect_ratio_prior_weight);
    problem.AddResidualBlock(aspect_left, nullptr, state.intrinsics_left.data());
    problem.AddResidualBlock(aspect_right, nullptr, state.intrinsics_right.data());
  }

  if (config.focal_prior_weight > 0.0) {
    ceres::CostFunction* focal_left = FocalPriorFactor::Create(
        state.init_intrinsics_left, config.focal_prior_weight);
    ceres::CostFunction* focal_right = FocalPriorFactor::Create(
        state.init_intrinsics_right, config.focal_prior_weight);
    problem.AddResidualBlock(focal_left, nullptr, state.intrinsics_left.data());
    problem.AddResidualBlock(focal_right, nullptr, state.intrinsics_right.data());
  }

  // Fix selected intrinsic parameters.
  // SubsetManifold 会只固定指定下标，其余内参仍可被优化。
  std::vector<int> fixed_intrinsic_indices;
  if (config.fix_principal_point) {
    fixed_intrinsic_indices.push_back(2);  // cx
    fixed_intrinsic_indices.push_back(3);  // cy
  }
  if (config.fix_distortion) {
    fixed_intrinsic_indices.push_back(4);  // k1
    fixed_intrinsic_indices.push_back(5);  // k2
    fixed_intrinsic_indices.push_back(6);  // p1
    fixed_intrinsic_indices.push_back(7);  // p2
    fixed_intrinsic_indices.push_back(8);  // k3
  }

  if (config.fix_camera_params) {
    // per-frame correction 使用这个分支：固定双目相机本身，只校正帧位姿/点。
    // 全局 BA 通常不会打开 fix_camera_params，因此相机内外参会和帧位姿一起优化。
    if (problem.HasParameterBlock(state.intrinsics_left.data())) {
      problem.SetParameterBlockConstant(state.intrinsics_left.data());
    }
    if (problem.HasParameterBlock(state.intrinsics_right.data())) {
      problem.SetParameterBlockConstant(state.intrinsics_right.data());
    }
    if (problem.HasParameterBlock(state.extrinsics.data())) {
      problem.SetParameterBlockConstant(state.extrinsics.data());
    }
  }

  if (config.fix_track_points) {
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      if (problem.HasParameterBlock(tracks[ti].point3d.data())) {
        problem.SetParameterBlockConstant(tracks[ti].point3d.data());
      }
    }
  }

  // Set intrinsics manifold and bounds.
  // 焦距使用相对初值的上下界，畸变使用固定数值范围，避免优化走到
  // 明显不合理的相机模型。
  auto set_intrinsics_bounds = [&](double* intr, double init_fx, double init_fy) {
    problem.SetParameterLowerBound(intr, 0, config.focal_lower_scale * init_fx);
    problem.SetParameterUpperBound(intr, 0, config.focal_upper_scale * init_fx);
    problem.SetParameterLowerBound(intr, 1, config.focal_lower_scale * init_fy);
    problem.SetParameterUpperBound(intr, 1, config.focal_upper_scale * init_fy);
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

  if (!config.fix_camera_params && problem.HasParameterBlock(state.intrinsics_left.data())) {
    SetSubsetParameterBlock(problem, state.intrinsics_left.data(), 9, fixed_intrinsic_indices);
    set_intrinsics_bounds(state.intrinsics_left.data(),
                          state.init_intrinsics_left[0], state.init_intrinsics_left[1]);
  }
  if (!config.fix_camera_params && problem.HasParameterBlock(state.intrinsics_right.data())) {
    SetSubsetParameterBlock(problem, state.intrinsics_right.data(), 9, fixed_intrinsic_indices);
    set_intrinsics_bounds(state.intrinsics_right.data(),
                          state.init_intrinsics_right[0], state.init_intrinsics_right[1]);
  }

  // Fix frame poses as needed.
  // frame_to_optimize >= 0：局部校正，仅目标帧位姿可变；
  // frame_to_optimize < 0：全局 BA，固定参考帧和非 active 帧，其余 active 帧可变。
  // 固定参考帧是为了去掉世界坐标系的任意刚体变换自由度；否则所有帧和点
  // 可以整体旋转/平移而不改变重投影误差，问题会欠约束。
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    const bool has_rvec = problem.HasParameterBlock(frames[fi].rvec.data());
    const bool has_tvec = problem.HasParameterBlock(frames[fi].tvec.data());
    if (!has_rvec && !has_tvec) {
      continue;
    }

    bool should_fix = false;
    if (frame_to_optimize >= 0) {
      should_fix = (static_cast<int>(fi) != frame_to_optimize);
    } else {
      should_fix = (static_cast<int>(fi) == state.fixed_frame_idx ||
                    fi >= active_frames.size() ||
                    !active_frames[fi]);
    }

    if (should_fix) {
      if (has_rvec) {
        problem.SetParameterBlockConstant(frames[fi].rvec.data());
      }
      if (has_tvec) {
        problem.SetParameterBlockConstant(frames[fi].tvec.data());
      }
    }
  }

  // Configure solver.
  // SPARSE_SCHUR 是 BA 的常用线性求解器，利用“相机/位姿 - 3D 点”
  // 的 Schur complement 稀疏结构。
  ceres::Solver::Options options;
  options.max_num_iterations = std::max(1, config.max_iterations);
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 24;
  options.minimizer_progress_to_stdout = true;

  // Solve
  ceres::Solve(options, &problem, &result.summary);

  // Compute RMSE.
  // Ceres cost = 1/2 * sum(residual^2)，所以这里乘 2 再除以 residual 维度数。
  if (result.summary.num_residuals <= 0) {
    result.init_rmse = 0.0;
    result.final_rmse = 0.0;
  } else {
    result.init_rmse = std::sqrt(2.0 * result.summary.initial_cost / result.summary.num_residuals);
    result.final_rmse = std::sqrt(2.0 * result.summary.final_cost / result.summary.num_residuals);
  }

  result.success = true;
  result.num_residuals = static_cast<int>(active_residuals);
  return result;
}

}  // namespace stereocalib
