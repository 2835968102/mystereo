#include "offline_stereo_ba.h"

#include <ceres/ceres.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <opencv2/calib3d.hpp>

#include "stereo_eval.h"
#include "stereo_factors.h"
#include "stereo_io.h"
#include "track_builder.h"

namespace stereocalib {
namespace {

cv::Mat ToRotation(const std::vector<double>& rvec)
{
  const cv::Mat rv = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]);
  cv::Mat R;
  cv::Rodrigues(rv, R);
  return R;
}

}  // namespace

// ─── Constructor ────────────────────────────────────────────────────────────

OfflineStereoBA::OfflineStereoBA(const OfflineBAInput& input, const Options& options)
    : input_(input), options_(options)
{
  intrinsics_left_ = input_.init_camera.left.ToVector();
  intrinsics_right_ = input_.init_camera.right.ToVector();
  extrinsics_ = input_.init_camera.extrinsics.ToVector();
  init_extrinsics_ = extrinsics_;
}

// ─── Ground truth ───────────────────────────────────────────────────────────

void OfflineStereoBA::SetGroundTruth(const StereoCamera& gt)
{
  has_ground_truth_ = true;
  ground_truth_ = gt;
}

void OfflineStereoBA::LoadFramePoses(const nlohmann::json& poses_json)
{
  frame_poses_json_ = poses_json;
}

void OfflineStereoBA::ApplyFramePosesToFrames()
{
  if (frame_poses_json_.empty() || !frame_poses_json_.contains("frames")) {
    return;
  }

  const nlohmann::json& frames_json = frame_poses_json_.at("frames");
  if (!frames_json.is_array()) {
    return;
  }

  // Build a map from image name to pose data
  std::map<std::string, nlohmann::json> left_image_to_pose;
  for (const auto& frame_json : frames_json) {
    if (!frame_json.contains("left_image") || !frame_json.contains("left_pose")) {
      continue;
    }
    std::string left_image = frame_json.at("left_image").get<std::string>();
    left_image_to_pose[left_image] = frame_json.at("left_pose");
  }

  // Apply ground truth poses to frames
  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    FrameState& frame = frames_[fi];

    // Find the left image name for this frame
    if (frame.left_image_idx < 0 || frame.left_image_idx >= static_cast<int>(images_.size())) {
      continue;
    }

    const std::string& left_image_name = images_[frame.left_image_idx].name;
    auto it = left_image_to_pose.find(left_image_name);
    if (it == left_image_to_pose.end()) {
      continue;
    }

    const nlohmann::json& pose_json = it->second;
    if (!pose_json.contains("R") || !pose_json.contains("t")) {
      continue;
    }

    // Extract R and t
    const std::vector<double> R_vec = pose_json.at("R").get<std::vector<double>>();
    const std::vector<double> t_vec = pose_json.at("t").get<std::vector<double>>();

    if (R_vec.size() != 9 || t_vec.size() != 3) {
      continue;
    }

    // Convert rotation matrix to Rodrigues vector
    cv::Mat R = (cv::Mat_<double>(3, 3) << R_vec[0], R_vec[1], R_vec[2],
                                            R_vec[3], R_vec[4], R_vec[5],
                                            R_vec[6], R_vec[7], R_vec[8]);
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    frame.has_gt_pose = true;
    frame.gt_rvec = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
    frame.gt_tvec = {t_vec[0], t_vec[1], t_vec[2]};
  }

  int num_frames_with_gt = 0;
  for (const auto& frame : frames_) {
    if (frame.has_gt_pose) {
      num_frames_with_gt++;
    }
  }

  std::cout << "Loaded ground truth poses for " << num_frames_with_gt
            << " / " << frames_.size() << " frames." << std::endl;
}

// ─── RunBundleAdjustment ────────────────────────────────────────────────────

bool OfflineStereoBA::RunBundleAdjustment(const std::vector<char>& active_frames,
                                          int max_num_iterations,
                                          ceres::Solver::Summary& summary,
                                          double& init_rmse,
                                          double& final_rmse,
                                          int frame_to_optimize)
{
  ceres::Problem problem;
  size_t active_residuals = 0;

  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    const Track& track = tracks_[ti];

    int active_obs = 0;
    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames_.size())) {
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

    for (size_t oi = 0; oi < track.observations.size(); ++oi) {
      const TrackObservation& obs = track.observations[oi];
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames_.size())) {
        continue;
      }
      if (obs.frame_idx >= static_cast<int>(active_frames.size()) || !active_frames[obs.frame_idx]) {
        continue;
      }

      ceres::CostFunction* cost = TrackReprojFactor::Create(obs.px, obs.is_left);
      ceres::LossFunction* loss = new ceres::HuberLoss(options_.huber_delta);
      problem.AddResidualBlock(cost,
                               loss,
                               intrinsics_left_.data(),
                               intrinsics_right_.data(),
                               extrinsics_.data(),
                               frames_[obs.frame_idx].rvec.data(),
                               tracks_[ti].point3d.data());
      active_residuals++;
    }
  }

  if (active_residuals == 0) {
    return false;
  }

  if (options_.baseline_prior_weight > 0.0) {
    ceres::CostFunction* prior_cost = BaselinePriorFactor::Create(init_extrinsics_, options_.baseline_prior_weight);
    problem.AddResidualBlock(prior_cost, NULL, extrinsics_.data());
  }
  if (options_.aspect_ratio_prior_weight > 0.0) {
    ceres::CostFunction* aspect_left = AspectRatioPriorFactor::Create(options_.aspect_ratio_prior_weight);
    ceres::CostFunction* aspect_right = AspectRatioPriorFactor::Create(options_.aspect_ratio_prior_weight);
    problem.AddResidualBlock(aspect_left, NULL, intrinsics_left_.data());
    problem.AddResidualBlock(aspect_right, NULL, intrinsics_right_.data());
  }

  std::vector<int> fixed_intrinsic_indices;
  fixed_intrinsic_indices.push_back(2);
  fixed_intrinsic_indices.push_back(3);
  if (options_.fix_distortion) {
    fixed_intrinsic_indices.push_back(4);
    fixed_intrinsic_indices.push_back(5);
    fixed_intrinsic_indices.push_back(6);
    fixed_intrinsic_indices.push_back(7);
    fixed_intrinsic_indices.push_back(8);
  }

  auto set_intrinsics_bounds = [&](double* intr, const Intrinsics& init_intr) {
    problem.SetParameterLowerBound(intr, 0, 0.5 * init_intr.fx);
    problem.SetParameterUpperBound(intr, 0, 1.5 * init_intr.fx);
    problem.SetParameterLowerBound(intr, 1, 0.5 * init_intr.fy);
    problem.SetParameterUpperBound(intr, 1, 1.5 * init_intr.fy);
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

  if (problem.HasParameterBlock(intrinsics_left_.data())) {
    problem.SetManifold(intrinsics_left_.data(), new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(intrinsics_left_.data(), input_.init_camera.left);
  }
  if (problem.HasParameterBlock(intrinsics_right_.data())) {
    problem.SetManifold(intrinsics_right_.data(), new ceres::SubsetManifold(9, fixed_intrinsic_indices));
    set_intrinsics_bounds(intrinsics_right_.data(), input_.init_camera.right);
  }

  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    if (!problem.HasParameterBlock(frames_[fi].rvec.data())) {
      continue;
    }

    bool should_fix = false;

    if (frame_to_optimize >= 0) {
      should_fix = (static_cast<int>(fi) != frame_to_optimize);
    } else {
      should_fix = (static_cast<int>(fi) == fixed_frame_idx_ ||
                    fi >= active_frames.size() ||
                    !active_frames[fi]);
    }

    if (should_fix) {
      problem.SetParameterBlockConstant(frames_[fi].rvec.data());
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = std::max(1, max_num_iterations);
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 4;
  options.minimizer_progress_to_stdout = true;

  ceres::Solve(options, &problem, &summary);

  if (summary.num_residuals <= 0) {
    init_rmse = 0.0;
    final_rmse = 0.0;
  } else {
    init_rmse = std::sqrt(2.0 * summary.initial_cost / summary.num_residuals);
    final_rmse = std::sqrt(2.0 * summary.final_cost / summary.num_residuals);
  }

  return true;
}

// ─── RejectOutliers ─────────────────────────────────────────────────────────

int OfflineStereoBA::RejectOutliers(double threshold)
{
  const double thresh2 = threshold * threshold;
  int rejected_count = 0;

  const cv::Mat rvec_rl = (cv::Mat_<double>(3, 1) << extrinsics_[0], extrinsics_[1], extrinsics_[2]);
  cv::Mat R_rl;
  cv::Rodrigues(rvec_rl, R_rl);
  const cv::Mat t_rl = (cv::Mat_<double>(3, 1) << extrinsics_[3], extrinsics_[4], extrinsics_[5]);

  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    Track& track = tracks_[ti];
    const cv::Mat X_w = (cv::Mat_<double>(3, 1) << track.point3d[0], track.point3d[1], track.point3d[2]);

    for (TrackObservation& obs : track.observations) {
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames_.size())) continue;

      const cv::Mat X_l = ToRotation(frames_[obs.frame_idx].rvec) * X_w;

      cv::Mat X_cam = X_l;
      const double* intr = intrinsics_left_.data();
      if (!obs.is_left) {
        X_cam = R_rl * X_l + t_rl;
        intr = intrinsics_right_.data();
      }

      const double Z = X_cam.at<double>(2, 0);
      if (Z <= 0.0) {
        obs.rejected = true;
        ++rejected_count;
        continue;
      }

      double u, v;
      ApplyDistAndProject(intr, X_cam.at<double>(0, 0) / Z, X_cam.at<double>(1, 0) / Z, u, v);

      const double du = obs.px.x - u;
      const double dv = obs.px.y - v;
      if (du * du + dv * dv > thresh2) {
        obs.rejected = true;
        ++rejected_count;
      }
    }
  }

  std::cout << "[Outlier Rejection] Rejected " << rejected_count
            << " observations above " << threshold << " px threshold" << std::endl;
  return rejected_count;
}

// ─── ApplyResult ────────────────────────────────────────────────────────────

void OfflineStereoBA::ApplyResult(StereoCamera& result)
{
  result.left.FromVector(intrinsics_left_);
  result.right.FromVector(intrinsics_right_);
  result.extrinsics.FromVector(extrinsics_);
}

// ─── PrintCurrentVsGroundTruth ──────────────────────────────────────────────

void OfflineStereoBA::PrintCurrentVsGroundTruth(const std::string& stage_name) const
{
  if (!has_ground_truth_) {
    return;
  }

  StereoCamera current;
  current.left.FromVector(intrinsics_left_);
  current.right.FromVector(intrinsics_right_);
  current.extrinsics.FromVector(extrinsics_);

  std::cout << "\n========== " << stage_name << " - Comparison with Ground Truth ==========\n";
  std::cout << std::showpos << std::fixed << std::setprecision(6);

  std::cout << "Left camera:\n";
  std::cout << "  fx: " << current.left.fx << " (gt: " << ground_truth_.left.fx
            << ", diff: " << (current.left.fx - ground_truth_.left.fx) << ")\n";
  std::cout << "  fy: " << current.left.fy << " (gt: " << ground_truth_.left.fy
            << ", diff: " << (current.left.fy - ground_truth_.left.fy) << ")\n";
  std::cout << "  cx: " << current.left.cx << " (gt: " << ground_truth_.left.cx
            << ", diff: " << (current.left.cx - ground_truth_.left.cx) << ")\n";
  std::cout << "  cy: " << current.left.cy << " (gt: " << ground_truth_.left.cy
            << ", diff: " << (current.left.cy - ground_truth_.left.cy) << ")\n";

  std::cout << "Right camera:\n";
  std::cout << "  fx: " << current.right.fx << " (gt: " << ground_truth_.right.fx
            << ", diff: " << (current.right.fx - ground_truth_.right.fx) << ")\n";
  std::cout << "  fy: " << current.right.fy << " (gt: " << ground_truth_.right.fy
            << ", diff: " << (current.right.fy - ground_truth_.right.fy) << ")\n";
  std::cout << "  cx: " << current.right.cx << " (gt: " << ground_truth_.right.cx
            << ", diff: " << (current.right.cx - ground_truth_.right.cx) << ")\n";
  std::cout << "  cy: " << current.right.cy << " (gt: " << ground_truth_.right.cy
            << ", diff: " << (current.right.cy - ground_truth_.right.cy) << ")\n";

  const double tx_diff = current.extrinsics.t.at<double>(0, 0) - ground_truth_.extrinsics.t.at<double>(0, 0);
  const double ty_diff = current.extrinsics.t.at<double>(1, 0) - ground_truth_.extrinsics.t.at<double>(1, 0);
  const double tz_diff = current.extrinsics.t.at<double>(2, 0) - ground_truth_.extrinsics.t.at<double>(2, 0);

  std::cout << "Extrinsics:\n";
  std::cout << "  t: [" << current.extrinsics.t.at<double>(0, 0) << ", "
            << current.extrinsics.t.at<double>(1, 0) << ", "
            << current.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  gt: [" << ground_truth_.extrinsics.t.at<double>(0, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(1, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  diff: [" << tx_diff << ", " << ty_diff << ", " << tz_diff << "]\n";

  const double baseline_current = TranslationNorm(current.extrinsics.t);
  const double baseline_gt = TranslationNorm(ground_truth_.extrinsics.t);

  std::cout << "  baseline: " << baseline_current << " (gt: " << baseline_gt
            << ", diff: " << (baseline_current - baseline_gt) << ")\n";

  const double rot_error_deg = RotationErrorDeg(current.extrinsics.R, ground_truth_.extrinsics.R);

  std::cout << "  rotation_error: " << rot_error_deg << " degrees\n";

  // ── Per-frame pose errors ─────────────────────────────────────────────────
  int num_frames_with_gt = 0;
  double total_rot_error = 0.0;
  double max_rot_error = 0.0;

  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    if (!frames_[fi].has_gt_pose) {
      continue;
    }
    num_frames_with_gt++;

    // Convert optimized rvec to rotation matrix
    cv::Mat opt_rvec = (cv::Mat_<double>(3, 1) << frames_[fi].rvec[0], frames_[fi].rvec[1], frames_[fi].rvec[2]);
    cv::Mat opt_R;
    cv::Rodrigues(opt_rvec, opt_R);

    // Convert GT rvec to rotation matrix
    cv::Mat gt_rvec = (cv::Mat_<double>(3, 1) << frames_[fi].gt_rvec[0], frames_[fi].gt_rvec[1], frames_[fi].gt_rvec[2]);
    cv::Mat gt_R;
    cv::Rodrigues(gt_rvec, gt_R);

    // Rotation error in degrees
    double frame_rot_error = RotationErrorDeg(opt_R, gt_R);

    total_rot_error += frame_rot_error;
    max_rot_error = std::max(max_rot_error, frame_rot_error);
  }

  if (num_frames_with_gt > 0) {
    std::cout << "\nPer-frame pose errors (with GT):\n";
    std::cout << "  Frames with GT: " << num_frames_with_gt << " / " << frames_.size() << "\n";
    std::cout << "  Avg rotation error: " << (total_rot_error / num_frames_with_gt) << " degrees\n";
    std::cout << "  Max rotation error: " << max_rot_error << " degrees\n";
  }

  std::cout << std::noshowpos;
  std::cout << "================================================================\n\n";
}

// ─── RecordOptimizationStage ────────────────────────────────────────────────

void OfflineStereoBA::RecordOptimizationStage(const std::string& stage_name, double reproj_error)
{
  using json = nlohmann::json;

  StereoCamera current;
  current.left.FromVector(intrinsics_left_);
  current.right.FromVector(intrinsics_right_);
  current.extrinsics.FromVector(extrinsics_);

  json stage_record;
  stage_record["stage"] = stage_name;
  stage_record["reproj_error"] = reproj_error;

  stage_record["camera"]["left"] = IntrinsicsToJson(current.left);
  stage_record["camera"]["right"] = IntrinsicsToJson(current.right);
  stage_record["camera"]["extrinsics"] = ExtrinsicsToJson(current.extrinsics);

  if (has_ground_truth_) {
    stage_record["diff_vs_gt"]["left"] = IntrinsicsDiffToJson(current.left, ground_truth_.left);
    stage_record["diff_vs_gt"]["right"] = IntrinsicsDiffToJson(current.right, ground_truth_.right);
    stage_record["diff_vs_gt"]["extrinsics"] = ExtrinsicsDiffToJson(current.extrinsics, ground_truth_.extrinsics);
  }

  // Add per-frame pose errors
  json frame_errors = json::array();
  double total_rot_error = 0.0;
  int num_frames_with_gt = 0;

  for (size_t fi = 0; fi < frames_.size(); ++fi) {
    if (!frames_[fi].has_gt_pose) {
      continue;
    }
    num_frames_with_gt++;

    // Convert optimized rvec to rotation matrix
    cv::Mat opt_rvec = (cv::Mat_<double>(3, 1) << frames_[fi].rvec[0], frames_[fi].rvec[1], frames_[fi].rvec[2]);
    cv::Mat opt_R;
    cv::Rodrigues(opt_rvec, opt_R);

    // Convert GT rvec to rotation matrix
    cv::Mat gt_rvec = (cv::Mat_<double>(3, 1) << frames_[fi].gt_rvec[0], frames_[fi].gt_rvec[1], frames_[fi].gt_rvec[2]);
    cv::Mat gt_R;
    cv::Rodrigues(gt_rvec, gt_R);

    // Rotation error in degrees
    double frame_rot_error = RotationErrorDeg(opt_R, gt_R);
    total_rot_error += frame_rot_error;

    json frame_error;
    frame_error["frame_id"] = frames_[fi].frame_id;
    frame_error["frame_idx"] = fi;
    frame_error["rotation_error_deg"] = frame_rot_error;
    frame_errors.push_back(frame_error);
  }

  if (num_frames_with_gt > 0) {
    stage_record["frame_pose_errors"] = frame_errors;
    stage_record["avg_frame_rotation_error_deg"] = total_rot_error / num_frames_with_gt;
  }

  optimization_history_.push_back(stage_record);
}

// ─── Solve ──────────────────────────────────────────────────────────────────

bool OfflineStereoBA::Solve(StereoCamera& result)
{
  // ── Track building ──────────────────────────────────────────────────────
  TrackBuildResult build_result;
  if (!BuildTracks(input_.pairs,
                   options_.max_match_score,
                   options_.min_pair_inliers,
                   options_.min_pair_inlier_ratio,
                   options_.min_track_len,
                   build_result)) {
    return false;
  }

  tracks_.swap(build_result.tracks);
  frames_.swap(build_result.frames);
  frame_ids_.swap(build_result.frame_ids);
  images_.swap(build_result.images);
  num_tracks_ = build_result.num_tracks;
  num_observations_ = build_result.num_observations;

  // ── Apply ground truth frame poses if available ─────────────────────────
  ApplyFramePosesToFrames();

  // ── Frame rotation initialization ───────────────────────────────────────
  std::vector<int> registration_order;
  if (!InitializeFrameRotations(input_.init_camera, tracks_, frames_,
                                 registration_order, fixed_frame_idx_)) {
    return false;
  }

  // ── Track point initialization ──────────────────────────────────────────
  if (!InitializeTrackPoints(input_.init_camera, extrinsics_, frames_, tracks_)) {
    return false;
  }

  if (registration_order.empty()) {
    return false;
  }

  // ── Incremental BA ────────────────────────────────────────────────────
  std::vector<char> active_frames(frames_.size(), 0);
  active_frames[registration_order[0]] = 1;

  bool have_rmse = false;
  int successful_registrations = 0;

  for (size_t i = 1; i < registration_order.size(); ++i) {
    const int frame_idx = registration_order[i];
    if (frame_idx < 0 || frame_idx >= static_cast<int>(active_frames.size())) {
      continue;
    }
    active_frames[frame_idx] = 1;
    successful_registrations++;

    ceres::Solver::Summary incremental_summary;
    double step_init_rmse = 0.0;
    double step_final_rmse = 0.0;
    if (RunBundleAdjustment(active_frames,
                            options_.incremental_max_iter,
                            incremental_summary,
                            step_init_rmse,
                            step_final_rmse,
                            frame_idx)) {
      if (!have_rmse) {
        init_reproj_error_ = step_init_rmse;
        have_rmse = true;
      }
      final_reproj_error_ = step_final_rmse;
      std::cout << "[Incremental BA] registered_frames=" << (i + 1)
                << "/" << registration_order.size()
                << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                << step_final_rmse << " px" << std::endl;

      std::string stage_name = "Incremental BA - Frame " + std::to_string(i + 1);
      PrintCurrentVsGroundTruth(stage_name);
      RecordOptimizationStage(stage_name, step_final_rmse);
    }

    if (options_.global_opt_interval > 0 &&
        successful_registrations % options_.global_opt_interval == 0) {
      ceres::Solver::Summary periodic_summary;
      double global_init_rmse = 0.0;
      double global_final_rmse = 0.0;
      if (RunBundleAdjustment(active_frames,
                              options_.max_iter,
                              periodic_summary,
                              global_init_rmse,
                              global_final_rmse)) {
        if (!have_rmse) {
          init_reproj_error_ = global_init_rmse;
          have_rmse = true;
        }
        final_reproj_error_ = global_final_rmse;
        std::cout << "[Global BA] registered_frames=" << (i + 1)
                  << "/" << registration_order.size()
                  << ", reproj_rmse=" << std::fixed << std::setprecision(4)
                  << global_final_rmse << " px" << std::endl;

        std::string stage_name = "Periodic Global BA - Frame " + std::to_string(i + 1);
        PrintCurrentVsGroundTruth(stage_name);
        RecordOptimizationStage(stage_name, global_final_rmse);
      }
    }
  }

  // ── Final global BA ───────────────────────────────────────────────────
  std::fill(active_frames.begin(), active_frames.end(), 1);
  if (!RunBundleAdjustment(active_frames,
                           options_.max_iter,
                           summary_,
                           init_reproj_error_,
                           final_reproj_error_)) {
    return false;
  }

  std::cout << summary_.BriefReport() << std::endl;
  std::cout << "Tracks=" << num_tracks_ << ", observations=" << num_observations_ << ", frames=" << frames_.size()
            << std::endl;
  std::cout << "Reprojection error: init=" << std::fixed << std::setprecision(4) << init_reproj_error_
            << " px, final=" << final_reproj_error_ << " px" << std::endl;

  PrintCurrentVsGroundTruth("Final Global BA");
  RecordOptimizationStage("Final Global BA", final_reproj_error_);

  const bool converged = (summary_.termination_type == ceres::CONVERGENCE ||
                          summary_.termination_type == ceres::NO_CONVERGENCE);
  const bool pass_reproj = (final_reproj_error_ <= options_.max_reproj_error);

  if (!converged) {
    std::cerr << "Offline BA did not converge." << std::endl;
  }
  if (!pass_reproj) {
    std::cerr << "Final reprojection error " << final_reproj_error_
              << " px exceeds threshold " << options_.max_reproj_error << " px." << std::endl;
  }

  const bool pass_fov = CheckFov(intrinsics_left_,  "Left camera") &&
                        CheckFov(intrinsics_right_, "Right camera");

  ApplyResult(result);
  return converged && pass_reproj && pass_fov;
}

}  // namespace stereocalib
