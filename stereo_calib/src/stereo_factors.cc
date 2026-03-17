#include "stereo_factors.h"

#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>

using namespace std;

namespace stereocalib {

// ─── Helper ──────────────────────────────────────────────────────────────────

// intr: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
void ApplyDistAndProject(const double* intr, double xn, double yn, double& u, double& v)
{
  const double fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
  const double k1 = intr[4], k2 = intr[5], p1 = intr[6], p2 = intr[7], k3 = intr[8];

  const double x  = xn, y = yn;
  const double r2 = x * x + y * y;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;

  // Radial + tangential distortion (OpenCV model)
  const double radial     = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_distorted = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y);

  u = fx * x_distorted + cx;
  v = fy * y_distorted + cy;
}

// ─── LeftReprojFactor ─────────────────────────────────────────────────────────

bool LeftReprojFactor::operator()(const double* intrinsics, const double* point3d, double* residual) const
{
  const double Z = point3d[2];
  if (Z <= 0.0) {
    // Point is behind the camera – apply a large penalty so the optimiser
    // avoids this configuration (same approach as PTZRayDistFactor).
    residual[0] = residual[1] = 1e6;
    return true;
  }

  const double xn = point3d[0] / Z;
  const double yn = point3d[1] / Z;

  double u, v;
  ApplyDistAndProject(intrinsics, xn, yn, u, v);

  residual[0] = obs_.x - u;
  residual[1] = obs_.y - v;
  return true;
}

// NumericDiff signature:  <Factor, method, residual_dim, param1_dim, param2_dim>
//   intrinsics [9], point3d [3]
ceres::CostFunction* LeftReprojFactor::Create(const cv::Point2f& obs)
{
  return new ceres::NumericDiffCostFunction<LeftReprojFactor, ceres::CENTRAL, 2, 9, 3>(new LeftReprojFactor(obs));
}

// ─── RightReprojFactor ────────────────────────────────────────────────────────

bool RightReprojFactor::operator()(const double* intrinsics, const double* extrinsics, const double* point3d,
                                   double* residual) const
{
  // Recover R_rl, t_rl from the extrinsics parameter block
  const cv::Mat rvec = (cv::Mat_<double>(3, 1) << extrinsics[0], extrinsics[1], extrinsics[2]);
  cv::Mat R_rl;
  cv::Rodrigues(rvec, R_rl);

  // Transform 3D point from left frame to right frame: X_r = R_rl * X_l + t_rl
  const cv::Mat pt_l = (cv::Mat_<double>(3, 1) << point3d[0], point3d[1], point3d[2]);
  const cv::Mat t_rl = (cv::Mat_<double>(3, 1) << extrinsics[3], extrinsics[4], extrinsics[5]);
  const cv::Mat pt_r = R_rl * pt_l + t_rl;

  const double Z = pt_r.at<double>(2, 0);
  if (Z <= 0.0) {
    residual[0] = residual[1] = 1e6;
    return true;
  }

  const double xn = pt_r.at<double>(0, 0) / Z;
  const double yn = pt_r.at<double>(1, 0) / Z;

  double u, v;
  ApplyDistAndProject(intrinsics, xn, yn, u, v);

  residual[0] = obs_.x - u;
  residual[1] = obs_.y - v;
  return true;
}

// NumericDiff signature:
//   intrinsics_right [9], extrinsics [6], point3d [3]
ceres::CostFunction* RightReprojFactor::Create(const cv::Point2f& obs)
{
  return new ceres::NumericDiffCostFunction<RightReprojFactor, ceres::CENTRAL, 2, 9, 6, 3>(new RightReprojFactor(obs));
}

// ─── TrackReprojFactor ──────────────────────────────────────────────────────

bool TrackReprojFactor::operator()(const double* intr_left,
                                   const double* intr_right,
                                   const double* extrinsics,
                                   const double* frame_rvec,
                                   const double* point3d,
                                   double* residual) const
{
  const cv::Mat rvec_lw = (cv::Mat_<double>(3, 1) << frame_rvec[0], frame_rvec[1], frame_rvec[2]);
  cv::Mat R_lw;
  cv::Rodrigues(rvec_lw, R_lw);

  const cv::Mat X_w = (cv::Mat_<double>(3, 1) << point3d[0], point3d[1], point3d[2]);
  const cv::Mat X_l = R_lw * X_w;

  cv::Mat X_cam = X_l;
  const double* intr = intr_left;

  if (!is_left_) {
    const cv::Mat rvec_rl = (cv::Mat_<double>(3, 1) << extrinsics[0], extrinsics[1], extrinsics[2]);
    cv::Mat R_rl;
    cv::Rodrigues(rvec_rl, R_rl);
    const cv::Mat t_rl = (cv::Mat_<double>(3, 1) << extrinsics[3], extrinsics[4], extrinsics[5]);
    X_cam = R_rl * X_l + t_rl;
    intr = intr_right;
  }

  const double Z = X_cam.at<double>(2, 0);
  if (Z <= 0.0) {
    residual[0] = 1e6;
    residual[1] = 1e6;
    return true;
  }

  const double xn = X_cam.at<double>(0, 0) / Z;
  const double yn = X_cam.at<double>(1, 0) / Z;

  double u = 0.0;
  double v = 0.0;
  ApplyDistAndProject(intr, xn, yn, u, v);

  residual[0] = static_cast<double>(obs_.x) - u;
  residual[1] = static_cast<double>(obs_.y) - v;
  return true;
}

// NumericDiff signature:
//   intr_left [9], intr_right [9], extrinsics [6], frame_rvec [3], point3d [3]
ceres::CostFunction* TrackReprojFactor::Create(const cv::Point2f& obs, bool is_left)
{
  return new ceres::NumericDiffCostFunction<TrackReprojFactor, ceres::CENTRAL, 2, 9, 9, 6, 3, 3>(
      new TrackReprojFactor(obs, is_left));
}

// ─── BaselinePriorFactor ────────────────────────────────────────────────────

BaselinePriorFactor::BaselinePriorFactor(const vector<double>& init_extrinsics, double weight)
    : init_t_(3, 0.0), weight_(weight)
{
  init_t_[0] = init_extrinsics[3];
  init_t_[1] = init_extrinsics[4];
  init_t_[2] = init_extrinsics[5];
}

bool BaselinePriorFactor::operator()(const double* extrinsics, double* residual) const
{
  residual[0] = weight_ * (extrinsics[3] - init_t_[0]);
  residual[1] = weight_ * (extrinsics[4] - init_t_[1]);
  residual[2] = weight_ * (extrinsics[5] - init_t_[2]);
  return true;
}

ceres::CostFunction* BaselinePriorFactor::Create(const vector<double>& init_extrinsics, double weight)
{
  return new ceres::NumericDiffCostFunction<BaselinePriorFactor, ceres::CENTRAL, 3, 6>(
      new BaselinePriorFactor(init_extrinsics, weight));
}

// ─── AspectRatioPriorFactor ─────────────────────────────────────────────────

bool AspectRatioPriorFactor::operator()(const double* intrinsics, double* residual) const
{
  residual[0] = weight_ * (intrinsics[0] - intrinsics[1]);  // fx - fy
  return true;
}

ceres::CostFunction* AspectRatioPriorFactor::Create(double weight)
{
  return new ceres::NumericDiffCostFunction<AspectRatioPriorFactor, ceres::CENTRAL, 1, 9>(
      new AspectRatioPriorFactor(weight));
}

}  // namespace stereocalib
