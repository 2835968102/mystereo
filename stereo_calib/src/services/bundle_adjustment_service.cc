#include "services/bundle_adjustment_service.h"

#include <cmath>

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

  // Add track reprojection residuals
  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    const Track& track = tracks[ti];

    // Count active observations
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

    // Add residuals for each observation
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
                               tracks[ti].point3d.data());
      active_residuals++;
    }
  }

  if (active_residuals == 0) {
    return result;
  }

  // Add priors
  if (config.baseline_prior_weight > 0.0) {
    ceres::CostFunction* prior_cost = BaselinePriorFactor::Create(
        state.init_extrinsics, config.baseline_prior_weight);
    problem.AddResidualBlock(prior_cost, nullptr, state.extrinsics.data());
  }
  
  if (config.aspect_ratio_prior_weight > 0.0) {
    ceres::CostFunction* aspect_left = AspectRatioPriorFactor::Create(
        config.aspect_ratio_prior_weight);
    ceres::CostFunction* aspect_right = AspectRatioPriorFactor::Create(
        config.aspect_ratio_prior_weight);
    problem.AddResidualBlock(aspect_left, nullptr, state.intrinsics_left.data());
    problem.AddResidualBlock(aspect_right, nullptr, state.intrinsics_right.data());
  }

  // Fix intrinsic parameters (cx, cy, and optionally distortion)
  std::vector<int> fixed_intrinsic_indices;
  fixed_intrinsic_indices.push_back(2);  // cx
  fixed_intrinsic_indices.push_back(3);  // cy
  if (config.fix_distortion) {
    fixed_intrinsic_indices.push_back(4);  // k1
    fixed_intrinsic_indices.push_back(5);  // k2
    fixed_intrinsic_indices.push_back(6);  // p1
    fixed_intrinsic_indices.push_back(7);  // p2
    fixed_intrinsic_indices.push_back(8);  // k3
  }

  // Set intrinsics manifold and bounds
  auto set_intrinsics_bounds = [&](double* intr, double init_fx, double init_fy) {
    problem.SetParameterLowerBound(intr, 0, 0.5 * init_fx);
    problem.SetParameterUpperBound(intr, 0, 1.5 * init_fx);
    problem.SetParameterLowerBound(intr, 1, 0.5 * init_fy);
    problem.SetParameterUpperBound(intr, 1, 1.5 * init_fy);
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

  if (problem.HasParameterBlock(state.intrinsics_left.data())) {
    problem.SetManifold(state.intrinsics_left.data(),
                        new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(state.intrinsics_left.data(),
                          state.intrinsics_left[0], state.intrinsics_left[1]);
  }
  if (problem.HasParameterBlock(state.intrinsics_right.data())) {
    problem.SetManifold(state.intrinsics_right.data(),
                        new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(state.intrinsics_right.data(),
                          state.intrinsics_right[0], state.intrinsics_right[1]);
  }

  // Fix frame poses as needed
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (!problem.HasParameterBlock(frames[fi].rvec.data())) {
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
      problem.SetParameterBlockConstant(frames[fi].rvec.data());
    }
  }

  // Configure solver
  ceres::Solver::Options options;
  options.max_num_iterations = std::max(1, config.max_iterations);
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 32;
  options.minimizer_progress_to_stdout = true;

  // Solve
  ceres::Solve(options, &problem, &result.summary);

  // Compute RMSE
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
