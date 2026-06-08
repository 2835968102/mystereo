#include "services/bundle_adjustment_service.h"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "stereo_factors.h"

namespace stereocalib {

namespace {

bool CameraCenterWorldFromFrame(const FrameState& frame, cv::Vec3d& center) {
  if (frame.rvec.size() < 3 || frame.tvec.size() < 3) {
    return false;
  }
  const cv::Mat rvec = (cv::Mat_<double>(3, 1)
      << frame.rvec[0], frame.rvec[1], frame.rvec[2]);
  cv::Mat R_lw;
  cv::Rodrigues(rvec, R_lw);
  const cv::Mat t_lw = (cv::Mat_<double>(3, 1)
      << frame.tvec[0], frame.tvec[1], frame.tvec[2]);
  const cv::Mat c = -R_lw.t() * t_lw;
  if (!cv::checkRange(c)) {
    return false;
  }
  center = cv::Vec3d(c.at<double>(0, 0), c.at<double>(1, 0), c.at<double>(2, 0));
  return true;
}

struct AlignedFramePositionTargets {
  std::vector<cv::Vec3d> centers;
  std::vector<char> valid;
  size_t count = 0;
};

struct AlignedFrameRotationTargets {
  std::vector<cv::Vec3d> rvecs;
  std::vector<char> valid;
  size_t count = 0;
};

struct AlignedAbsoluteFrameRotationTargets {
  std::vector<cv::Vec3d> rvecs;
  std::vector<char> valid;
  size_t count = 0;
};

bool FrameRotationMatrix(const FrameState& frame, cv::Mat& R) {
  if (frame.rvec.size() < 3) {
    return false;
  }
  const cv::Mat rvec = (cv::Mat_<double>(3, 1)
      << frame.rvec[0], frame.rvec[1], frame.rvec[2]);
  cv::Rodrigues(rvec, R);
  return cv::checkRange(R);
}

bool FrameTargetRotationMatrix(const FrameState& frame, cv::Mat& R) {
  if (!frame.has_gt_pose || frame.gt_rvec.size() < 3) {
    return false;
  }
  const cv::Mat rvec = (cv::Mat_<double>(3, 1)
      << frame.gt_rvec[0], frame.gt_rvec[1], frame.gt_rvec[2]);
  cv::Rodrigues(rvec, R);
  return cv::checkRange(R);
}

cv::Vec3d RotationVectorFromMatrix(const cv::Mat& R) {
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  return cv::Vec3d(rvec.at<double>(0, 0),
                   rvec.at<double>(1, 0),
                   rvec.at<double>(2, 0));
}

bool SameFrameSequence(const FrameState& a, const FrameState& b) {
  return a.sequence_id.empty() ||
         b.sequence_id.empty() ||
         a.sequence_id == b.sequence_id;
}

double ComputeReprojectionRmse(const BAState& state,
                               const std::vector<FrameState>& frames,
                               const std::vector<Track>& tracks,
                               const std::vector<char>& active_frames) {
  double squared_error_sum = 0.0;
  size_t residual_dims = 0;
  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    const Track& track = tracks[ti];
    if (track.point3d.size() < 3) {
      continue;
    }

    int active_obs = 0;
    for (const TrackObservation& obs : track.observations) {
      if (obs.rejected) {
        continue;
      }
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) ||
          !active_frames[obs.frame_idx]) {
        continue;
      }
      active_obs++;
    }
    if (active_obs < 2) {
      continue;
    }

    for (const TrackObservation& obs : track.observations) {
      if (obs.rejected) {
        continue;
      }
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) ||
          !active_frames[obs.frame_idx]) {
        continue;
      }

      const FrameState& frame = frames[obs.frame_idx];
      if (frame.rvec.size() < 3 || frame.tvec.size() < 3 ||
          state.intrinsics_left.size() < 9 ||
          state.intrinsics_right.size() < 9 ||
          state.extrinsics.size() < 6) {
        continue;
      }

      double residual[2] = {0.0, 0.0};
      TrackReprojFactor factor(obs.px, obs.is_left);
      factor(state.intrinsics_left.data(),
             state.intrinsics_right.data(),
             state.extrinsics.data(),
             frame.rvec.data(),
             frame.tvec.data(),
             track.point3d.data(),
             residual);
      if (!std::isfinite(residual[0]) || !std::isfinite(residual[1])) {
        continue;
      }
      squared_error_sum += residual[0] * residual[0] + residual[1] * residual[1];
      residual_dims += 2;
    }
  }

  if (residual_dims == 0) {
    return 0.0;
  }
  return std::sqrt(squared_error_sum / static_cast<double>(residual_dims));
}

AlignedFramePositionTargets EstimateAlignedFramePositionTargets(
    const std::vector<FrameState>& frames,
    const std::vector<char>& active_frames) {
  AlignedFramePositionTargets targets;
  targets.centers.assign(frames.size(), cv::Vec3d(0.0, 0.0, 0.0));
  targets.valid.assign(frames.size(), 0);

  std::vector<cv::Vec3d> current_centers;
  std::vector<cv::Vec3d> oxts_centers;
  std::vector<size_t> frame_indices;
  current_centers.reserve(frames.size());
  oxts_centers.reserve(frames.size());
  frame_indices.reserve(frames.size());

  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (fi >= active_frames.size() || !active_frames[fi]) {
      continue;
    }
    if (!frames[fi].has_gt_pose || frames[fi].gt_tvec.size() < 3) {
      continue;
    }
    cv::Vec3d current_center;
    if (!CameraCenterWorldFromFrame(frames[fi], current_center)) {
      continue;
    }
    current_centers.push_back(current_center);
    oxts_centers.emplace_back(
        frames[fi].gt_tvec[0],
        frames[fi].gt_tvec[1],
        frames[fi].gt_tvec[2]);
    frame_indices.push_back(fi);
  }

  if (current_centers.size() < 4) {
    return targets;
  }

  cv::Vec3d mean_current(0.0, 0.0, 0.0);
  cv::Vec3d mean_oxts(0.0, 0.0, 0.0);
  for (size_t i = 0; i < current_centers.size(); ++i) {
    mean_current += current_centers[i];
    mean_oxts += oxts_centers[i];
  }
  const double n = static_cast<double>(current_centers.size());
  mean_current *= 1.0 / n;
  mean_oxts *= 1.0 / n;

  cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
  double variance_oxts = 0.0;
  for (size_t i = 0; i < current_centers.size(); ++i) {
    const cv::Vec3d x = oxts_centers[i] - mean_oxts;
    const cv::Vec3d y = current_centers[i] - mean_current;
    variance_oxts += x.dot(x) / n;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        covariance.at<double>(row, col) += y[row] * x[col] / n;
      }
    }
  }
  constexpr double kMinOxtsRmsSpreadMeters = 0.5;
  if (variance_oxts <= kMinOxtsRmsSpreadMeters * kMinOxtsRmsSpreadMeters ||
      !cv::checkRange(covariance)) {
    return targets;
  }

  cv::SVD svd(covariance, cv::SVD::FULL_UV);
  cv::Mat D = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat R = svd.u * svd.vt;
  if (cv::determinant(R) < 0.0) {
    D.at<double>(2, 2) = -1.0;
    R = svd.u * D * svd.vt;
  }

  double scale_numerator = 0.0;
  for (int i = 0; i < 3; ++i) {
    scale_numerator += D.at<double>(i, i) * svd.w.at<double>(i, 0);
  }
  const double scale = scale_numerator / variance_oxts;
  if (!std::isfinite(scale) || scale <= 0.0) {
    return targets;
  }

  const cv::Mat mean_current_mat = (cv::Mat_<double>(3, 1)
      << mean_current[0], mean_current[1], mean_current[2]);
  const cv::Mat mean_oxts_mat = (cv::Mat_<double>(3, 1)
      << mean_oxts[0], mean_oxts[1], mean_oxts[2]);
  const cv::Mat translation = mean_current_mat - scale * R * mean_oxts_mat;
  if (!cv::checkRange(R) || !cv::checkRange(translation)) {
    return targets;
  }

  for (size_t i = 0; i < frame_indices.size(); ++i) {
    const cv::Vec3d& oxts = oxts_centers[i];
    const cv::Mat p = (cv::Mat_<double>(3, 1) << oxts[0], oxts[1], oxts[2]);
    const cv::Mat aligned = scale * R * p + translation;
    const size_t fi = frame_indices[i];
    targets.centers[fi] = cv::Vec3d(
        aligned.at<double>(0, 0),
        aligned.at<double>(1, 0),
        aligned.at<double>(2, 0));
    targets.valid[fi] = 1;
    targets.count++;
  }
  return targets;
}

AlignedFrameRotationTargets EstimateAlignedFrameRotationTargets(
    const std::vector<FrameState>& frames,
    const std::vector<char>& active_frames,
    int min_stride,
    int max_stride) {
  AlignedFrameRotationTargets targets;
  targets.rvecs.assign(frames.size() * frames.size(), cv::Vec3d(0.0, 0.0, 0.0));
  targets.valid.assign(frames.size() * frames.size(), 0);
  if (frames.empty()) {
    return targets;
  }

  const int first_stride = std::max(1, min_stride);
  const int last_stride = std::max(first_stride, max_stride);
  constexpr double kMinTargetRotationRad = 1e-4;
  for (int stride = first_stride; stride <= last_stride; ++stride) {
    for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
      const size_t fj = fi + static_cast<size_t>(stride);
      if (fi >= active_frames.size() || fj >= active_frames.size() ||
          !active_frames[fi] || !active_frames[fj]) {
        continue;
      }
      if (!SameFrameSequence(frames[fi], frames[fj])) {
        continue;
      }
      cv::Mat R_current_i;
      cv::Mat R_current_j;
      cv::Mat R_oxts_i;
      cv::Mat R_oxts_j;
      if (!FrameRotationMatrix(frames[fi], R_current_i) ||
          !FrameRotationMatrix(frames[fj], R_current_j) ||
          !FrameTargetRotationMatrix(frames[fi], R_oxts_i) ||
          !FrameTargetRotationMatrix(frames[fj], R_oxts_j)) {
        continue;
      }

      const cv::Vec3d oxts_rvec =
          RotationVectorFromMatrix(R_oxts_j * R_oxts_i.t());
      if (cv::norm(oxts_rvec) < kMinTargetRotationRad) {
        continue;
      }
      const size_t index = fi * frames.size() + fj;
      targets.rvecs[index] = oxts_rvec;
      targets.valid[index] = 1;
      targets.count++;
    }
  }
  return targets;
}

AlignedAbsoluteFrameRotationTargets EstimateAlignedAbsoluteFrameRotationTargets(
    const std::vector<FrameState>& frames,
    const std::vector<char>& active_frames,
    int anchor_frame_idx) {
  AlignedAbsoluteFrameRotationTargets targets;
  targets.rvecs.assign(frames.size(), cv::Vec3d(0.0, 0.0, 0.0));
  targets.valid.assign(frames.size(), 0);
  if (frames.empty() ||
      anchor_frame_idx < 0 ||
      anchor_frame_idx >= static_cast<int>(frames.size()) ||
      anchor_frame_idx >= static_cast<int>(active_frames.size()) ||
      !active_frames[anchor_frame_idx]) {
    return targets;
  }

  cv::Mat R_current_anchor;
  cv::Mat R_target_anchor;
  if (!FrameRotationMatrix(frames[anchor_frame_idx], R_current_anchor) ||
      !FrameTargetRotationMatrix(frames[anchor_frame_idx], R_target_anchor)) {
    return targets;
  }
  // Frame rotations are world-to-camera. A BA gauge rotation therefore
  // right-multiplies all frame rotations, so align external absolute
  // rotations through the fixed anchor on the right.
  const cv::Mat R_gauge = R_target_anchor.t() * R_current_anchor;
  if (!cv::checkRange(R_gauge)) {
    return targets;
  }

  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (fi >= active_frames.size() || !active_frames[fi]) {
      continue;
    }
    cv::Mat R_target;
    if (!FrameTargetRotationMatrix(frames[fi], R_target)) {
      continue;
    }
    const cv::Mat R_aligned = R_target * R_gauge;
    if (!cv::checkRange(R_aligned)) {
      continue;
    }
    targets.rvecs[fi] = RotationVectorFromMatrix(R_aligned);
    targets.valid[fi] = 1;
    targets.count++;
  }
  return targets;
}

}  // namespace

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
  const double initial_reprojection_rmse =
      ComputeReprojectionRmse(state, frames, tracks, active_frames);

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

  if (config.focal_mean_prior_weight > 0.0 &&
      state.init_intrinsics_left.size() >= 2 &&
      state.init_intrinsics_right.size() >= 2) {
    const double target_focal =
        0.25 * (state.init_intrinsics_left[0] +
                state.init_intrinsics_left[1] +
                state.init_intrinsics_right[0] +
                state.init_intrinsics_right[1]);
    ceres::CostFunction* focal_mean_left = FocalMeanPriorFactor::Create(
        target_focal, config.focal_mean_prior_weight);
    ceres::CostFunction* focal_mean_right = FocalMeanPriorFactor::Create(
        target_focal, config.focal_mean_prior_weight);
    problem.AddResidualBlock(focal_mean_left, nullptr, state.intrinsics_left.data());
    problem.AddResidualBlock(focal_mean_right, nullptr, state.intrinsics_right.data());
  }

  if (config.stereo_intrinsics_consistency_weight > 0.0) {
    ceres::CostFunction* stereo_intrinsics =
        StereoIntrinsicsConsistencyFactor::Create(config.stereo_intrinsics_consistency_weight);
    problem.AddResidualBlock(stereo_intrinsics,
                             nullptr,
                             state.intrinsics_left.data(),
                             state.intrinsics_right.data());
  }

  if (config.principal_point_mean_prior_weight > 0.0 &&
      state.init_intrinsics_left.size() >= 4 &&
      state.init_intrinsics_right.size() >= 4) {
    const double target_cx =
        0.5 * (state.init_intrinsics_left[2] + state.init_intrinsics_right[2]);
    const double target_cy =
        0.5 * (state.init_intrinsics_left[3] + state.init_intrinsics_right[3]);
    ceres::CostFunction* pp_left = PrincipalPointMeanPriorFactor::Create(
        target_cx, target_cy, config.principal_point_mean_prior_weight);
    ceres::CostFunction* pp_right = PrincipalPointMeanPriorFactor::Create(
        target_cx, target_cy, config.principal_point_mean_prior_weight);
    problem.AddResidualBlock(pp_left, nullptr, state.intrinsics_left.data());
    problem.AddResidualBlock(pp_right, nullptr, state.intrinsics_right.data());
  }

  if (config.frame_distance_prior_weight > 0.0 && frame_to_optimize < 0) {
    const int min_stride = std::max(1, config.frame_distance_prior_stride);
    const int max_stride = std::max(min_stride, config.frame_distance_prior_max_stride);
    const double stride_count = static_cast<double>(max_stride - min_stride + 1);
    const double per_stride_weight =
        config.frame_distance_prior_weight / std::sqrt(std::max(1.0, stride_count));
    for (int stride = min_stride; stride <= max_stride; ++stride) {
      for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
        const size_t fj = fi + static_cast<size_t>(stride);
        if (fi >= active_frames.size() || fj >= active_frames.size() ||
            !active_frames[fi] || !active_frames[fj]) {
          continue;
        }
        if (!SameFrameSequence(frames[fi], frames[fj])) {
          continue;
        }
        if (!frames[fi].has_gt_pose || !frames[fj].has_gt_pose ||
            frames[fi].gt_tvec.size() < 3 || frames[fj].gt_tvec.size() < 3) {
          continue;
        }
        const double dx = frames[fi].gt_tvec[0] - frames[fj].gt_tvec[0];
        const double dy = frames[fi].gt_tvec[1] - frames[fj].gt_tvec[1];
        const double dz = frames[fi].gt_tvec[2] - frames[fj].gt_tvec[2];
        const double target_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        constexpr double kMinFrameDistancePriorMeters = 0.2;
        if (target_distance < kMinFrameDistancePriorMeters) {
          continue;
        }
        ceres::CostFunction* distance_prior = FrameDistancePriorFactor::Create(
            target_distance, per_stride_weight);
        problem.AddResidualBlock(distance_prior,
                                 nullptr,
                                 frames[fi].rvec.data(),
                                 frames[fi].tvec.data(),
                                 frames[fj].rvec.data(),
                                 frames[fj].tvec.data());
      }
    }
  }

  if (config.frame_position_prior_weight > 0.0 && frame_to_optimize < 0) {
    const AlignedFramePositionTargets position_targets =
        EstimateAlignedFramePositionTargets(frames, active_frames);
    if (position_targets.count >= 4) {
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        if (fi >= active_frames.size() || !active_frames[fi] ||
            fi >= position_targets.valid.size() ||
            !position_targets.valid[fi]) {
          continue;
        }
        ceres::CostFunction* position_prior = FramePositionPriorFactor::Create(
            position_targets.centers[fi], config.frame_position_prior_weight);
        problem.AddResidualBlock(position_prior,
                                 nullptr,
                                 frames[fi].rvec.data(),
                                 frames[fi].tvec.data());
      }
    }
  }

  if (config.frame_translation_vector_prior_weight > 0.0 && frame_to_optimize < 0) {
    const int min_stride = std::max(1, config.frame_distance_prior_stride);
    const int max_stride = std::max(min_stride, config.frame_distance_prior_max_stride);
    const double stride_count = static_cast<double>(max_stride - min_stride + 1);
    const double per_stride_weight =
        config.frame_translation_vector_prior_weight /
        std::sqrt(std::max(1.0, stride_count));
    for (int stride = min_stride; stride <= max_stride; ++stride) {
      for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
        const size_t fj = fi + static_cast<size_t>(stride);
        if (fi >= active_frames.size() || fj >= active_frames.size() ||
            !active_frames[fi] || !active_frames[fj]) {
          continue;
        }
        if (!SameFrameSequence(frames[fi], frames[fj])) {
          continue;
        }
        if (!frames[fi].has_gt_pose || !frames[fj].has_gt_pose ||
            frames[fi].gt_tvec.size() < 3 || frames[fj].gt_tvec.size() < 3 ||
            frames[fi].gt_rvec.size() < 3) {
          continue;
        }
        cv::Mat R_target_i;
        if (!FrameTargetRotationMatrix(frames[fi], R_target_i)) {
          continue;
        }
        const cv::Mat target_delta_world = (cv::Mat_<double>(3, 1)
            << frames[fj].gt_tvec[0] - frames[fi].gt_tvec[0],
               frames[fj].gt_tvec[1] - frames[fi].gt_tvec[1],
               frames[fj].gt_tvec[2] - frames[fi].gt_tvec[2]);
        constexpr double kMinFrameTranslationPriorMeters = 0.2;
        if (cv::norm(target_delta_world) < kMinFrameTranslationPriorMeters) {
          continue;
        }
        const cv::Mat target_delta_i = R_target_i * target_delta_world;
        if (!cv::checkRange(target_delta_i)) {
          continue;
        }
        const cv::Vec3d target_delta(
            target_delta_i.at<double>(0, 0),
            target_delta_i.at<double>(1, 0),
            target_delta_i.at<double>(2, 0));
        ceres::CostFunction* translation_prior =
            FrameTranslationVectorPriorFactor::Create(
                target_delta, per_stride_weight);
        problem.AddResidualBlock(translation_prior,
                                 nullptr,
                                 frames[fi].rvec.data(),
                                 frames[fi].tvec.data(),
                                 frames[fj].rvec.data(),
                                 frames[fj].tvec.data());
      }
    }
  }

  if (config.frame_translation_direction_prior_weight > 0.0 && frame_to_optimize < 0) {
    const int min_stride = std::max(1, config.frame_distance_prior_stride);
    const int max_stride = std::max(min_stride, config.frame_distance_prior_max_stride);
    const double stride_count = static_cast<double>(max_stride - min_stride + 1);
    const double per_stride_weight =
        config.frame_translation_direction_prior_weight /
        std::sqrt(std::max(1.0, stride_count));
    for (int stride = min_stride; stride <= max_stride; ++stride) {
      for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
        const size_t fj = fi + static_cast<size_t>(stride);
        if (fi >= active_frames.size() || fj >= active_frames.size() ||
            !active_frames[fi] || !active_frames[fj]) {
          continue;
        }
        if (!SameFrameSequence(frames[fi], frames[fj])) {
          continue;
        }
        if (!frames[fi].has_gt_pose || !frames[fj].has_gt_pose ||
            frames[fi].gt_tvec.size() < 3 || frames[fj].gt_tvec.size() < 3 ||
            frames[fi].gt_rvec.size() < 3) {
          continue;
        }
        cv::Mat R_target_i;
        if (!FrameTargetRotationMatrix(frames[fi], R_target_i)) {
          continue;
        }
        const cv::Mat target_delta_world = (cv::Mat_<double>(3, 1)
            << frames[fj].gt_tvec[0] - frames[fi].gt_tvec[0],
               frames[fj].gt_tvec[1] - frames[fi].gt_tvec[1],
               frames[fj].gt_tvec[2] - frames[fi].gt_tvec[2]);
        constexpr double kMinFrameTranslationPriorMeters = 0.2;
        if (cv::norm(target_delta_world) < kMinFrameTranslationPriorMeters) {
          continue;
        }
        const cv::Mat target_delta_i = R_target_i * target_delta_world;
        const double target_norm = cv::norm(target_delta_i);
        if (!cv::checkRange(target_delta_i) || target_norm < 1e-9) {
          continue;
        }
        const cv::Vec3d target_direction(
            target_delta_i.at<double>(0, 0) / target_norm,
            target_delta_i.at<double>(1, 0) / target_norm,
            target_delta_i.at<double>(2, 0) / target_norm);
        ceres::CostFunction* direction_prior =
            FrameTranslationDirectionPriorFactor::Create(
                target_direction, per_stride_weight);
        problem.AddResidualBlock(direction_prior,
                                 nullptr,
                                 frames[fi].rvec.data(),
                                 frames[fi].tvec.data(),
                                 frames[fj].rvec.data(),
                                 frames[fj].tvec.data());
      }
    }
  }

  if (config.frame_rotation_angle_prior_weight > 0.0 && frame_to_optimize < 0) {
    const int min_stride = std::max(1, config.frame_distance_prior_stride);
    const int max_stride = std::max(min_stride, config.frame_distance_prior_max_stride);
    const double stride_count = static_cast<double>(max_stride - min_stride + 1);
    const double per_stride_weight =
        config.frame_rotation_angle_prior_weight / std::sqrt(std::max(1.0, stride_count));
    for (int stride = min_stride; stride <= max_stride; ++stride) {
      for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
        const size_t fj = fi + static_cast<size_t>(stride);
        if (fi >= active_frames.size() || fj >= active_frames.size() ||
            !active_frames[fi] || !active_frames[fj]) {
          continue;
        }
        if (!SameFrameSequence(frames[fi], frames[fj])) {
          continue;
        }
        if (!frames[fi].has_gt_pose || !frames[fj].has_gt_pose ||
            frames[fi].gt_rvec.size() < 3 || frames[fj].gt_rvec.size() < 3) {
          continue;
        }
        const cv::Mat rvec_i = (cv::Mat_<double>(3, 1)
            << frames[fi].gt_rvec[0], frames[fi].gt_rvec[1], frames[fi].gt_rvec[2]);
        const cv::Mat rvec_j = (cv::Mat_<double>(3, 1)
            << frames[fj].gt_rvec[0], frames[fj].gt_rvec[1], frames[fj].gt_rvec[2]);
        cv::Mat R_i;
        cv::Mat R_j;
        cv::Rodrigues(rvec_i, R_i);
        cv::Rodrigues(rvec_j, R_j);
        cv::Mat rel_rvec;
        cv::Rodrigues(R_j * R_i.t(), rel_rvec);
        const double target_angle = cv::norm(rel_rvec);
        if (!std::isfinite(target_angle)) {
          continue;
        }
        ceres::CostFunction* rotation_prior = FrameRotationAnglePriorFactor::Create(
            target_angle, per_stride_weight);
        problem.AddResidualBlock(rotation_prior,
                                 nullptr,
                                 frames[fi].rvec.data(),
                                 frames[fj].rvec.data());
      }
    }
  }

  if (config.frame_rotation_vector_prior_weight > 0.0 && frame_to_optimize < 0) {
    const int min_stride = std::max(1, config.frame_distance_prior_stride);
    const int max_stride = std::max(min_stride, config.frame_distance_prior_max_stride);
    const AlignedFrameRotationTargets rotation_targets =
        EstimateAlignedFrameRotationTargets(frames, active_frames, min_stride, max_stride);
    if (rotation_targets.count >= 3) {
      const double stride_count = static_cast<double>(max_stride - min_stride + 1);
      const double per_stride_weight =
          config.frame_rotation_vector_prior_weight /
          std::sqrt(std::max(1.0, stride_count));
      for (int stride = min_stride; stride <= max_stride; ++stride) {
        for (size_t fi = 0; fi + static_cast<size_t>(stride) < frames.size(); ++fi) {
          const size_t fj = fi + static_cast<size_t>(stride);
          const size_t target_index = fi * frames.size() + fj;
          if (fi >= active_frames.size() || fj >= active_frames.size() ||
              !active_frames[fi] || !active_frames[fj] ||
              target_index >= rotation_targets.valid.size() ||
              !rotation_targets.valid[target_index]) {
            continue;
          }
          if (!SameFrameSequence(frames[fi], frames[fj])) {
            continue;
          }
          ceres::CostFunction* rotation_prior =
              FrameRotationVectorPriorFactor::Create(
                  rotation_targets.rvecs[target_index], per_stride_weight);
          problem.AddResidualBlock(rotation_prior,
                                   nullptr,
                                   frames[fi].rvec.data(),
                                   frames[fj].rvec.data());
        }
      }
    }
  }

  if (config.frame_absolute_rotation_prior_weight > 0.0 && frame_to_optimize < 0) {
    const AlignedAbsoluteFrameRotationTargets rotation_targets =
        EstimateAlignedAbsoluteFrameRotationTargets(
            frames, active_frames, state.fixed_frame_idx);
    if (rotation_targets.count >= 2) {
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        if (fi >= active_frames.size() || !active_frames[fi] ||
            fi >= rotation_targets.valid.size() ||
            !rotation_targets.valid[fi]) {
          continue;
        }
        ceres::CostFunction* rotation_prior =
            FrameAbsoluteRotationPriorFactor::Create(
                rotation_targets.rvecs[fi],
                config.frame_absolute_rotation_prior_weight);
        problem.AddResidualBlock(rotation_prior,
                                 nullptr,
                                 frames[fi].rvec.data());
      }
    }
  }

  // Fix selected intrinsic parameters.
  // SubsetManifold 会只固定指定下标，其余内参仍可被优化。
  std::vector<int> fixed_intrinsic_indices;
  if (config.fix_focal_length) {
    fixed_intrinsic_indices.push_back(0);  // fx
    fixed_intrinsic_indices.push_back(1);  // fy
  }
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
  } else if (config.fix_stereo_extrinsics &&
             problem.HasParameterBlock(state.extrinsics.data())) {
    problem.SetParameterBlockConstant(state.extrinsics.data());
  } else if ((config.fix_stereo_rotation || config.fix_stereo_yz_translation) &&
             problem.HasParameterBlock(state.extrinsics.data())) {
    std::vector<int> fixed_extrinsic_indices;
    if (config.fix_stereo_rotation) {
      fixed_extrinsic_indices.push_back(0);
      fixed_extrinsic_indices.push_back(1);
      fixed_extrinsic_indices.push_back(2);
    }
    if (config.fix_stereo_yz_translation) {
      fixed_extrinsic_indices.push_back(4);
      fixed_extrinsic_indices.push_back(5);
    }
    problem.SetManifold(state.extrinsics.data(),
                        new ceres::SubsetManifold(6, fixed_extrinsic_indices));
  }

  if (config.fix_track_points) {
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      if (problem.HasParameterBlock(tracks[ti].point3d.data())) {
        problem.SetParameterBlockConstant(tracks[ti].point3d.data());
      }
    }
  }

  if (config.stereo_tx_delta_bound > 0.0 &&
      problem.HasParameterBlock(state.extrinsics.data()) &&
      state.init_extrinsics.size() > 3) {
    const double tx_bound = std::abs(config.stereo_tx_delta_bound);
    problem.SetParameterLowerBound(
        state.extrinsics.data(), 3, state.init_extrinsics[3] - tx_bound);
    problem.SetParameterUpperBound(
        state.extrinsics.data(), 3, state.init_extrinsics[3] + tx_bound);
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
    problem.SetManifold(state.intrinsics_left.data(),
                        new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(state.intrinsics_left.data(),
                          state.init_intrinsics_left[0], state.init_intrinsics_left[1]);
  }
  if (!config.fix_camera_params && problem.HasParameterBlock(state.intrinsics_right.data())) {
    problem.SetManifold(state.intrinsics_right.data(),
                        new ceres::SubsetManifold(9, fixed_intrinsic_indices));
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
      should_fix = config.fix_frame_poses ||
                   static_cast<int>(fi) == state.fixed_frame_idx ||
                   std::find(state.fixed_frame_indices.begin(),
                             state.fixed_frame_indices.end(),
                             static_cast<int>(fi)) != state.fixed_frame_indices.end() ||
                   fi >= active_frames.size() ||
                   !active_frames[fi];
    }

    if (should_fix) {
      if (has_rvec) {
        problem.SetParameterBlockConstant(frames[fi].rvec.data());
      }
      if (has_tvec) {
        problem.SetParameterBlockConstant(frames[fi].tvec.data());
      }
    } else {
      if (config.fix_frame_rotations && has_rvec) {
        problem.SetParameterBlockConstant(frames[fi].rvec.data());
      }
      if (config.fix_frame_translations && has_tvec) {
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

  // Report the image-domain RMSE only. Once motion/shape priors are present,
  // Ceres' total residual count is no longer a pure reprojection metric.
  if (result.summary.num_residuals <= 0) {
    result.init_rmse = 0.0;
    result.final_rmse = 0.0;
  } else {
    result.init_rmse = initial_reprojection_rmse;
    result.final_rmse = ComputeReprojectionRmse(state, frames, tracks, active_frames);
  }

  result.success = true;
  result.num_residuals = static_cast<int>(active_residuals);
  return result;
}

}  // namespace stereocalib
