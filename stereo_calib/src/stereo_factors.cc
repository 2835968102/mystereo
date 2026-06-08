#include "stereo_factors.h"

#include <ceres/ceres.h>
#include <cmath>
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
                                   const double* frame_tvec,
                                   const double* point3d,
                                   double* residual) const
{
  const cv::Mat rvec_lw = (cv::Mat_<double>(3, 1) << frame_rvec[0], frame_rvec[1], frame_rvec[2]);
  cv::Mat R_lw;
  cv::Rodrigues(rvec_lw, R_lw);

  const cv::Mat t_lw = (cv::Mat_<double>(3, 1) << frame_tvec[0], frame_tvec[1], frame_tvec[2]);
  const cv::Mat X_w = (cv::Mat_<double>(3, 1) << point3d[0], point3d[1], point3d[2]);
  const cv::Mat X_l = R_lw * X_w + t_lw;

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
//   intr_left [9], intr_right [9], extrinsics [6], frame_rvec [3], frame_tvec [3], point3d [3]
ceres::CostFunction* TrackReprojFactor::Create(const cv::Point2f& obs, bool is_left)
{
  return new ceres::NumericDiffCostFunction<TrackReprojFactor, ceres::CENTRAL, 2, 9, 9, 6, 3, 3, 3>(
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

// ─── TxPriorFactor ───────────────────────────────────────────────────────────

TxPriorFactor::TxPriorFactor(const vector<double>& init_extrinsics, double weight)
    : init_tx_(init_extrinsics[3]), weight_(weight)
{
}

bool TxPriorFactor::operator()(const double* extrinsics, double* residual) const
{
  residual[0] = weight_ * (extrinsics[3] - init_tx_);
  return true;
}

ceres::CostFunction* TxPriorFactor::Create(const vector<double>& init_extrinsics,
                                           double weight)
{
  return new ceres::NumericDiffCostFunction<TxPriorFactor, ceres::CENTRAL, 1, 6>(
      new TxPriorFactor(init_extrinsics, weight));
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

// ─── FocalPriorFactor ────────────────────────────────────────────────────────

FocalPriorFactor::FocalPriorFactor(const vector<double>& init_intrinsics, double weight)
    : init_fx_(init_intrinsics[0]), init_fy_(init_intrinsics[1]), weight_(weight)
{
}

bool FocalPriorFactor::operator()(const double* intrinsics, double* residual) const
{
  residual[0] = weight_ * (intrinsics[0] - init_fx_);
  residual[1] = weight_ * (intrinsics[1] - init_fy_);
  return true;
}

ceres::CostFunction* FocalPriorFactor::Create(const vector<double>& init_intrinsics,
                                              double weight)
{
  return new ceres::NumericDiffCostFunction<FocalPriorFactor, ceres::CENTRAL, 2, 9>(
      new FocalPriorFactor(init_intrinsics, weight));
}

// ─── FocalMeanPriorFactor ──────────────────────────────────────────────────

bool FocalMeanPriorFactor::operator()(const double* intrinsics,
                                      double* residual) const
{
  residual[0] = weight_ * (intrinsics[0] - target_focal_);
  residual[1] = weight_ * (intrinsics[1] - target_focal_);
  return true;
}

ceres::CostFunction* FocalMeanPriorFactor::Create(double target_focal,
                                                  double weight)
{
  return new ceres::NumericDiffCostFunction<
      FocalMeanPriorFactor, ceres::CENTRAL, 2, 9>(
      new FocalMeanPriorFactor(target_focal, weight));
}

// ─── StereoIntrinsicsConsistencyFactor ──────────────────────────────────────

bool StereoIntrinsicsConsistencyFactor::operator()(const double* intrinsics_left,
                                                   const double* intrinsics_right,
                                                   double* residual) const
{
  residual[0] = weight_ * (intrinsics_left[0] - intrinsics_right[0]);  // fx
  residual[1] = weight_ * (intrinsics_left[1] - intrinsics_right[1]);  // fy
  residual[2] = weight_ * (intrinsics_left[2] - intrinsics_right[2]);  // cx
  residual[3] = weight_ * (intrinsics_left[3] - intrinsics_right[3]);  // cy
  return true;
}

ceres::CostFunction* StereoIntrinsicsConsistencyFactor::Create(double weight)
{
  return new ceres::NumericDiffCostFunction<
      StereoIntrinsicsConsistencyFactor, ceres::CENTRAL, 4, 9, 9>(
      new StereoIntrinsicsConsistencyFactor(weight));
}

// ─── PrincipalPointMeanPriorFactor ──────────────────────────────────────────

bool PrincipalPointMeanPriorFactor::operator()(const double* intrinsics,
                                               double* residual) const
{
  residual[0] = weight_ * (intrinsics[2] - target_cx_);
  residual[1] = weight_ * (intrinsics[3] - target_cy_);
  return true;
}

ceres::CostFunction* PrincipalPointMeanPriorFactor::Create(double target_cx,
                                                           double target_cy,
                                                           double weight)
{
  return new ceres::NumericDiffCostFunction<
      PrincipalPointMeanPriorFactor, ceres::CENTRAL, 2, 9>(
      new PrincipalPointMeanPriorFactor(target_cx, target_cy, weight));
}

// ─── FrameDistancePriorFactor ──────────────────────────────────────────────

namespace {

cv::Mat CameraCenterFromWorldToCameraPose(const double* frame_rvec,
                                          const double* frame_tvec)
{
  const cv::Mat rvec = (cv::Mat_<double>(3, 1) << frame_rvec[0], frame_rvec[1], frame_rvec[2]);
  cv::Mat R_lw;
  cv::Rodrigues(rvec, R_lw);
  const cv::Mat t_lw = (cv::Mat_<double>(3, 1) << frame_tvec[0], frame_tvec[1], frame_tvec[2]);
  return -R_lw.t() * t_lw;
}

}  // namespace

bool FrameDistancePriorFactor::operator()(const double* frame_a_rvec,
                                           const double* frame_a_tvec,
                                           const double* frame_b_rvec,
                                           const double* frame_b_tvec,
                                           double* residual) const
{
  const cv::Mat center_a = CameraCenterFromWorldToCameraPose(frame_a_rvec, frame_a_tvec);
  const cv::Mat center_b = CameraCenterFromWorldToCameraPose(frame_b_rvec, frame_b_tvec);
  const double dx = center_a.at<double>(0, 0) - center_b.at<double>(0, 0);
  const double dy = center_a.at<double>(1, 0) - center_b.at<double>(1, 0);
  const double dz = center_a.at<double>(2, 0) - center_b.at<double>(2, 0);
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  residual[0] = weight_ * (distance - target_distance_);
  return true;
}

ceres::CostFunction* FrameDistancePriorFactor::Create(double target_distance,
                                                       double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameDistancePriorFactor, ceres::CENTRAL, 1, 3, 3, 3, 3>(
      new FrameDistancePriorFactor(target_distance, weight));
}

// ─── FramePositionPriorFactor ─────────────────────────────────────────────

bool FramePositionPriorFactor::operator()(const double* frame_rvec,
                                          const double* frame_tvec,
                                          double* residual) const
{
  const cv::Mat center = CameraCenterFromWorldToCameraPose(frame_rvec, frame_tvec);
  residual[0] = weight_ * (center.at<double>(0, 0) - target_center_[0]);
  residual[1] = weight_ * (center.at<double>(1, 0) - target_center_[1]);
  residual[2] = weight_ * (center.at<double>(2, 0) - target_center_[2]);
  return true;
}

ceres::CostFunction* FramePositionPriorFactor::Create(const cv::Vec3d& target_center,
                                                       double weight)
{
  return new ceres::NumericDiffCostFunction<
      FramePositionPriorFactor, ceres::CENTRAL, 3, 3, 3>(
      new FramePositionPriorFactor(target_center, weight));
}

// ─── FrameTranslationVectorPriorFactor ─────────────────────────────────────

bool FrameTranslationVectorPriorFactor::operator()(const double* frame_a_rvec,
                                                   const double* frame_a_tvec,
                                                   const double* frame_b_rvec,
                                                   const double* frame_b_tvec,
                                                   double* residual) const
{
  const cv::Mat rvec_a = (cv::Mat_<double>(3, 1)
      << frame_a_rvec[0], frame_a_rvec[1], frame_a_rvec[2]);
  cv::Mat R_a;
  cv::Rodrigues(rvec_a, R_a);
  const cv::Mat center_a =
      CameraCenterFromWorldToCameraPose(frame_a_rvec, frame_a_tvec);
  const cv::Mat center_b =
      CameraCenterFromWorldToCameraPose(frame_b_rvec, frame_b_tvec);
  const cv::Mat delta_a = R_a * (center_b - center_a);
  residual[0] = weight_ * (delta_a.at<double>(0, 0) - target_delta_[0]);
  residual[1] = weight_ * (delta_a.at<double>(1, 0) - target_delta_[1]);
  residual[2] = weight_ * (delta_a.at<double>(2, 0) - target_delta_[2]);
  return true;
}

ceres::CostFunction* FrameTranslationVectorPriorFactor::Create(
    const cv::Vec3d& target_delta,
    double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameTranslationVectorPriorFactor, ceres::CENTRAL, 3, 3, 3, 3, 3>(
      new FrameTranslationVectorPriorFactor(target_delta, weight));
}

// ─── FrameTranslationDirectionPriorFactor ──────────────────────────────────

bool FrameTranslationDirectionPriorFactor::operator()(const double* frame_a_rvec,
                                                      const double* frame_a_tvec,
                                                      const double* frame_b_rvec,
                                                      const double* frame_b_tvec,
                                                      double* residual) const
{
  const cv::Mat rvec_a = (cv::Mat_<double>(3, 1)
      << frame_a_rvec[0], frame_a_rvec[1], frame_a_rvec[2]);
  cv::Mat R_a;
  cv::Rodrigues(rvec_a, R_a);
  const cv::Mat center_a =
      CameraCenterFromWorldToCameraPose(frame_a_rvec, frame_a_tvec);
  const cv::Mat center_b =
      CameraCenterFromWorldToCameraPose(frame_b_rvec, frame_b_tvec);
  const cv::Mat delta_a = R_a * (center_b - center_a);
  const double norm = cv::norm(delta_a);
  constexpr double kMinNorm = 1e-9;
  if (!std::isfinite(norm) || norm < kMinNorm) {
    residual[0] = 0.0;
    residual[1] = 0.0;
    residual[2] = 0.0;
    return true;
  }
  residual[0] = weight_ * (delta_a.at<double>(0, 0) / norm - target_direction_[0]);
  residual[1] = weight_ * (delta_a.at<double>(1, 0) / norm - target_direction_[1]);
  residual[2] = weight_ * (delta_a.at<double>(2, 0) / norm - target_direction_[2]);
  return true;
}

ceres::CostFunction* FrameTranslationDirectionPriorFactor::Create(
    const cv::Vec3d& target_direction,
    double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameTranslationDirectionPriorFactor, ceres::CENTRAL, 3, 3, 3, 3, 3>(
      new FrameTranslationDirectionPriorFactor(target_direction, weight));
}

// ─── FrameRotationAnglePriorFactor ────────────────────────────────────────

bool FrameRotationAnglePriorFactor::operator()(const double* frame_a_rvec,
                                               const double* frame_b_rvec,
                                               double* residual) const
{
  const cv::Mat rvec_a = (cv::Mat_<double>(3, 1)
      << frame_a_rvec[0], frame_a_rvec[1], frame_a_rvec[2]);
  const cv::Mat rvec_b = (cv::Mat_<double>(3, 1)
      << frame_b_rvec[0], frame_b_rvec[1], frame_b_rvec[2]);
  cv::Mat R_a;
  cv::Mat R_b;
  cv::Rodrigues(rvec_a, R_a);
  cv::Rodrigues(rvec_b, R_b);
  const cv::Mat R_rel = R_b * R_a.t();
  cv::Mat rvec_rel;
  cv::Rodrigues(R_rel, rvec_rel);
  const double angle = cv::norm(rvec_rel);
  residual[0] = weight_ * (angle - target_angle_rad_);
  return true;
}

ceres::CostFunction* FrameRotationAnglePriorFactor::Create(double target_angle_rad,
                                                           double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameRotationAnglePriorFactor, ceres::CENTRAL, 1, 3, 3>(
      new FrameRotationAnglePriorFactor(target_angle_rad, weight));
}

// ─── FrameRotationVectorPriorFactor ────────────────────────────────────────

bool FrameRotationVectorPriorFactor::operator()(const double* frame_a_rvec,
                                                const double* frame_b_rvec,
                                                double* residual) const
{
  const cv::Mat rvec_a = (cv::Mat_<double>(3, 1)
      << frame_a_rvec[0], frame_a_rvec[1], frame_a_rvec[2]);
  const cv::Mat rvec_b = (cv::Mat_<double>(3, 1)
      << frame_b_rvec[0], frame_b_rvec[1], frame_b_rvec[2]);
  cv::Mat R_a;
  cv::Mat R_b;
  cv::Rodrigues(rvec_a, R_a);
  cv::Rodrigues(rvec_b, R_b);
  const cv::Mat R_rel = R_b * R_a.t();
  cv::Mat rvec_rel;
  cv::Rodrigues(R_rel, rvec_rel);
  residual[0] = weight_ * (rvec_rel.at<double>(0, 0) - target_rvec_[0]);
  residual[1] = weight_ * (rvec_rel.at<double>(1, 0) - target_rvec_[1]);
  residual[2] = weight_ * (rvec_rel.at<double>(2, 0) - target_rvec_[2]);
  return true;
}

ceres::CostFunction* FrameRotationVectorPriorFactor::Create(
    const cv::Vec3d& target_rvec,
    double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameRotationVectorPriorFactor, ceres::CENTRAL, 3, 3, 3>(
      new FrameRotationVectorPriorFactor(target_rvec, weight));
}

// ─── FrameAbsoluteRotationPriorFactor ──────────────────────────────────────

bool FrameAbsoluteRotationPriorFactor::operator()(const double* frame_rvec,
                                                  double* residual) const
{
  const cv::Mat rvec = (cv::Mat_<double>(3, 1)
      << frame_rvec[0], frame_rvec[1], frame_rvec[2]);
  const cv::Mat target_rvec = (cv::Mat_<double>(3, 1)
      << target_rvec_[0], target_rvec_[1], target_rvec_[2]);
  cv::Mat R;
  cv::Mat R_target;
  cv::Rodrigues(rvec, R);
  cv::Rodrigues(target_rvec, R_target);
  cv::Mat delta_rvec;
  cv::Rodrigues(R * R_target.t(), delta_rvec);
  residual[0] = weight_ * delta_rvec.at<double>(0, 0);
  residual[1] = weight_ * delta_rvec.at<double>(1, 0);
  residual[2] = weight_ * delta_rvec.at<double>(2, 0);
  return true;
}

ceres::CostFunction* FrameAbsoluteRotationPriorFactor::Create(
    const cv::Vec3d& target_rvec,
    double weight)
{
  return new ceres::NumericDiffCostFunction<
      FrameAbsoluteRotationPriorFactor, ceres::CENTRAL, 3, 3>(
      new FrameAbsoluteRotationPriorFactor(target_rvec, weight));
}

}  // namespace stereocalib
