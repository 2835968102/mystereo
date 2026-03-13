#include "stereo_optimizer.h"

#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>

#include "stereo_factors.h"

using namespace std;

namespace stereocalib {

// ─── Constructor ─────────────────────────────────────────────────────────────

StereoOptimizer::StereoOptimizer(const vector<StereoPair>& pairs, const StereoCamera& init_camera, int max_iter,
                                 double max_reproj_error)
    : pairs_(pairs), max_iter_(max_iter), max_reproj_error_(max_reproj_error)
{
  SetUpParams(init_camera);
  Triangulate(init_camera);
}

// ─── Parameter initialisation ────────────────────────────────────────────────

void StereoOptimizer::SetUpParams(const StereoCamera& camera)
{
  intrinsics_left_  = camera.left.ToVector();
  intrinsics_right_ = camera.right.ToVector();
  extrinsics_       = camera.extrinsics.ToVector();
}

// ─── Triangulation ───────────────────────────────────────────────────────────
//
// For each stereo match we need an initial 3D point in the left camera frame.
// We use OpenCV's cv::triangulatePoints after un-distorting both observations.
//
// Projection matrices:
//   P_l = I_{3×4}                   (normalised – after undistortPoints)
//   P_r = [R_rl | t_rl]             (normalised – after undistortPoints)

void StereoOptimizer::Triangulate(const StereoCamera& camera)
{
  // P_l (normalised coordinates, identity)
  cv::Mat P_l = cv::Mat::eye(3, 4, CV_64F);

  // P_r = [R_rl | t_rl]
  cv::Mat P_r(3, 4, CV_64F);
  camera.extrinsics.R.copyTo(P_r(cv::Range(0, 3), cv::Range(0, 3)));
  camera.extrinsics.t.copyTo(P_r(cv::Range(0, 3), cv::Range(3, 4)));

  points3d_.resize(pairs_.size());

  for (size_t i = 0; i < pairs_.size(); ++i) {
    const StereoPair& pair = pairs_[i];
    const size_t      N    = pair.matches.size();
    points3d_[i].assign(N, vector<double>(3, 0.0));

    if (N == 0)
      continue;

    // Collect raw pixel observations
    vector<cv::Point2f> pts_l(N), pts_r(N);
    for (size_t j = 0; j < N; ++j) {
      pts_l[j] = pair.matches[j].pt_left;
      pts_r[j] = pair.matches[j].pt_right;
    }

    // Undistort to normalised coordinates (output has no intrinsics applied)
    vector<cv::Point2f> pts_l_nd, pts_r_nd;
    cv::undistortPoints(pts_l, pts_l_nd, camera.left.K(),  camera.left.dist());
    cv::undistortPoints(pts_r, pts_r_nd, camera.right.K(), camera.right.dist());

    // Triangulate (homogeneous 4×N result in float)
    cv::Mat pts4d;
    cv::triangulatePoints(P_l, P_r, pts_l_nd, pts_r_nd, pts4d);

    for (size_t j = 0; j < N; ++j) {
      const float w = pts4d.at<float>(3, j);
      if (std::abs(w) < 1e-9f)
        continue;
      points3d_[i][j][0] = pts4d.at<float>(0, j) / w;
      points3d_[i][j][1] = pts4d.at<float>(1, j) / w;
      points3d_[i][j][2] = pts4d.at<float>(2, j) / w;
    }
  }
}

// ─── Build Ceres problem ─────────────────────────────────────────────────────

void StereoOptimizer::AddReprojConstraints()
{
  // Huber robust loss (δ = 1.0 pixel) – down-weights outlier matches
  // while still allowing them to be gradually corrected.
  const double kHuberDelta = 1.0;

  for (size_t i = 0; i < pairs_.size(); ++i) {
    const StereoPair& pair = pairs_[i];
    for (size_t j = 0; j < pair.matches.size(); ++j) {
      const StereoMatch& m  = pair.matches[j];
      auto&              pt = points3d_[i][j];

      // Skip points that triangulated behind either camera
      if (pt[2] <= 0.0)
        continue;

      // Left reprojection
      ceres::CostFunction* cost_l = LeftReprojFactor::Create(m.pt_left);
      problem_.AddResidualBlock(cost_l, new ceres::HuberLoss(kHuberDelta),
                                intrinsics_left_.data(), pt.data());

      // Right reprojection (adds extrinsics as an additional parameter block)
      ceres::CostFunction* cost_r = RightReprojFactor::Create(m.pt_right);
      problem_.AddResidualBlock(cost_r, new ceres::HuberLoss(kHuberDelta),
                                intrinsics_right_.data(), extrinsics_.data(), pt.data());
    }
  }

  // ── Fix principal points (cx, cy) ─────────────────────────────────────────
  // cx and cy at indices {2,3} are hard to recover from stereo correspondences
  // alone; fixing them avoids degeneracy.  Remove these lines if you have
  // enough baseline / known image centres.
  const vector<int> kFixedIntrinsicIndices = {2, 3};  // cx, cy
  if (problem_.HasParameterBlock(intrinsics_left_.data())) {
    problem_.SetManifold(intrinsics_left_.data(),
                         new ceres::SubsetManifold(9, kFixedIntrinsicIndices));
  }
  if (problem_.HasParameterBlock(intrinsics_right_.data())) {
    problem_.SetManifold(intrinsics_right_.data(),
                         new ceres::SubsetManifold(9, kFixedIntrinsicIndices));
  }
}

// ─── Solve ───────────────────────────────────────────────────────────────────

bool StereoOptimizer::Solve(StereoCamera& camera)
{
  AddReprojConstraints();

  ceres::Solver::Options options;
  options.max_num_iterations         = max_iter_;
  options.linear_solver_type         = ceres::SPARSE_SCHUR;  // efficient for BA
  options.minimizer_progress_to_stdout = true;
  options.num_threads                = 4;

  ceres::Solve(options, &problem_, &summary_);

  CalReprojError();

  cout << summary_.BriefReport() << endl;
  cout << "Reprojection error:  init = " << init_reproj_error_ << " px"
       << "  ->  final = " << final_reproj_error_ << " px" << endl;

  if (!CheckResults())
    return false;

  ObtainResults(camera);
  return true;
}

// ─── Post-solve helpers ───────────────────────────────────────────────────────

void StereoOptimizer::CalReprojError()
{
  if (summary_.num_residuals <= 0)
    return;
  // RMSE over all residual components (same formula as ptzray_optimizer.cc)
  init_reproj_error_  = std::sqrt(2.0 * summary_.initial_cost / summary_.num_residuals);
  final_reproj_error_ = std::sqrt(2.0 * summary_.final_cost   / summary_.num_residuals);
}

bool StereoOptimizer::CheckResults() const
{
  if (summary_.termination_type != ceres::TerminationType::CONVERGENCE) {
    cerr << "Optimisation did not converge." << endl;
    return false;
  }
  if (final_reproj_error_ > max_reproj_error_) {
    cerr << "Final reprojection error " << final_reproj_error_ << " px exceeds threshold " << max_reproj_error_ << " px." << endl;
    return false;
  }

  // ── FOV sanity check ──────────────────────────────────────────────────────
  // Use the principal point (cx) as a proxy for the half image width.
  // Horizontal half-FOV = atan(cx / fx).  Reject results outside [10°, 160°].
  const double kMinFovDeg = 10.0;
  const double kMaxFovDeg = 160.0;
  const double kDegPerRad = 180.0 / M_PI;

  auto check_fov = [&](const vector<double>& intr, const char* name) -> bool {
    const double fx = intr[0];
    const double cx = intr[2];
    if (fx <= 0.0 || cx <= 0.0) {
      cerr << name << " has non-positive fx or cx after optimisation." << endl;
      return false;
    }
    const double fov_deg = 2.0 * std::atan(cx / fx) * kDegPerRad;
    if (fov_deg < kMinFovDeg || fov_deg > kMaxFovDeg) {
      cerr << name << " estimated FOV " << fov_deg << " deg is outside ["
           << kMinFovDeg << ", " << kMaxFovDeg << "] deg." << endl;
      return false;
    }
    return true;
  };

  if (!check_fov(intrinsics_left_,  "Left camera") ||
      !check_fov(intrinsics_right_, "Right camera")) {
    return false;
  }

  return true;
}

void StereoOptimizer::ObtainResults(StereoCamera& camera)
{
  camera.left.FromVector(intrinsics_left_);
  camera.right.FromVector(intrinsics_right_);
  camera.extrinsics.FromVector(extrinsics_);
}

}  // namespace stereocalib
