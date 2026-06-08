/*
 * Ceres cost functions for stereo camera calibration.
 *
 * Factor types:
 *   LeftReprojFactor        – residual between observed left pixel and projected 3D point
 *   RightReprojFactor       – residual between observed right pixel and projected 3D point
 *   TrackReprojFactor       – reprojection with per-frame rotation (offline multi-frame BA)
 *   BaselinePriorFactor     – soft prior on stereo baseline translation
 *   TxPriorFactor           – soft prior on stereo x translation
 *   AspectRatioPriorFactor  – soft prior encouraging fx ≈ fy
 *   FocalPriorFactor        – soft prior keeping fx/fy near initial values
 *   FocalMeanPriorFactor    – soft prior on the common focal center
*   StereoIntrinsicsConsistencyFactor – soft prior tying left/right intrinsics
*   PrincipalPointMeanPriorFactor – soft prior on the rectified common center
*   FrameDistancePriorFactor – soft prior on metric distance between frames
*   FramePositionPriorFactor – soft prior on aligned external frame positions
*   FrameTranslationVectorPriorFactor – soft prior on relative frame translation
*   FrameTranslationDirectionPriorFactor – soft prior on relative motion direction
*   FrameRotationAnglePriorFactor – soft prior on relative rotation magnitude
*   FrameRotationVectorPriorFactor – soft prior on relative rotation vector
*   FrameAbsoluteRotationPriorFactor – soft prior on external absolute rotation
 *
 * All use NumericDiffCostFunction with CENTRAL differences (same pattern as
 * krt_optimizer / ptzray_optimizer in the parent project) so that OpenCV
 * routines (Rodrigues, etc.) can be called freely inside operator().
 */

#ifndef STEREO_CALIB_SRC_STEREO_FACTORS_H
#define STEREO_CALIB_SRC_STEREO_FACTORS_H

#include <ceres/ceres.h>
#include <opencv2/core.hpp>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

// ─── Helper ──────────────────────────────────────────────────────────────────

/**
 * Apply radial + tangential distortion and project into pixel coordinates.
 *
 * @param intr   Intrinsics parameter block [fx,fy,cx,cy,k1,k2,p1,p2,k3]
 * @param xn     Normalised image x  (X/Z)
 * @param yn     Normalised image y  (Y/Z)
 * @param u      Output pixel u
 * @param v      Output pixel v
 */
void ApplyDistAndProject(const double* intr, double xn, double yn, double& u, double& v);

// ─── Left reprojection factor ─────────────────────────────────────────────────
//
// Left camera is the reference frame, so no extrinsics are needed.
// Optimises: intrinsics_left [9], point3d [3]
// Residual dim: 2

class LeftReprojFactor {
 public:
  explicit LeftReprojFactor(const cv::Point2f& obs) : obs_(obs) {}

  bool operator()(const double* intrinsics, const double* point3d, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& obs);

 private:
  cv::Point2f obs_;
};

// ─── Right reprojection factor ────────────────────────────────────────────────
//
// Projects the same 3D point (in left frame) into the right camera using
//   X_right = R_rl * X_left + t_rl
// Optimises: intrinsics_right [9], extrinsics [6], point3d [3]
// Residual dim: 2

class RightReprojFactor {
 public:
  explicit RightReprojFactor(const cv::Point2f& obs) : obs_(obs) {}

  bool operator()(const double* intrinsics, const double* extrinsics, const double* point3d, double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& obs);

 private:
  cv::Point2f obs_;
};

// ─── Track reprojection factor (multi-frame BA) ──────────────────────────────
//
// Reprojects a world-frame 3D point through a per-frame pose into either
// the left or right camera.
// Optimises: intr_left [9], intr_right [9], extrinsics [6], frame_rvec [3], frame_tvec [3], point3d [3]
// Residual dim: 2

class TrackReprojFactor {
 public:
  TrackReprojFactor(const cv::Point2f& obs, bool is_left) : obs_(obs), is_left_(is_left) {}

  bool operator()(const double* intr_left,
                  const double* intr_right,
                  const double* extrinsics,
                  const double* frame_rvec,
                  const double* frame_tvec,
                  const double* point3d,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Point2f& obs, bool is_left);

 private:
  cv::Point2f obs_;
  bool is_left_ = true;
};

// ─── Baseline prior factor ───────────────────────────────────────────────────
//
// Penalises deviations of the stereo translation from initial values.
// Optimises: extrinsics [6]
// Residual dim: 3

class BaselinePriorFactor {
 public:
  BaselinePriorFactor(const std::vector<double>& init_extrinsics, double weight);

  bool operator()(const double* extrinsics, double* residual) const;

  static ceres::CostFunction* Create(const std::vector<double>& init_extrinsics, double weight);

 private:
  std::vector<double> init_t_;
  double weight_ = 0.0;
};

// ─── Tx prior factor ─────────────────────────────────────────────────────────
//
// Penalises deviations of stereo tx from the initial value.
// Optimises: extrinsics [6]
// Residual dim: 1

class TxPriorFactor {
 public:
  TxPriorFactor(const std::vector<double>& init_extrinsics, double weight);

  bool operator()(const double* extrinsics, double* residual) const;

  static ceres::CostFunction* Create(const std::vector<double>& init_extrinsics,
                                     double weight);

 private:
  double init_tx_ = 0.0;
  double weight_ = 0.0;
};

// ─── Aspect ratio prior factor ───────────────────────────────────────────────
//
// Penalises fx ≠ fy (encourages square pixels).
// Optimises: intrinsics [9]
// Residual dim: 1

class AspectRatioPriorFactor {
 public:
  explicit AspectRatioPriorFactor(double weight) : weight_(weight) {}

  bool operator()(const double* intrinsics, double* residual) const;

  static ceres::CostFunction* Create(double weight);

 private:
  double weight_ = 0.0;
};

// ─── Focal prior factor ──────────────────────────────────────────────────────
//
// Penalises deviations of fx/fy from initial values.
// Optimises: intrinsics [9]
// Residual dim: 2

class FocalPriorFactor {
 public:
  FocalPriorFactor(const std::vector<double>& init_intrinsics, double weight);

  bool operator()(const double* intrinsics, double* residual) const;

  static ceres::CostFunction* Create(const std::vector<double>& init_intrinsics,
                                     double weight);

 private:
  double init_fx_ = 0.0;
  double init_fy_ = 0.0;
  double weight_ = 0.0;
};

// ─── Focal mean prior factor ────────────────────────────────────────────────
//
// Penalises fx/fy drift away from a common focal value estimated from the
// initial stereo calibration itself. This anchors the shared focal scale
// without using ground truth.
// Optimises: intrinsics [9]
// Residual dim: 2

class FocalMeanPriorFactor {
 public:
  FocalMeanPriorFactor(double target_focal, double weight)
      : target_focal_(target_focal), weight_(weight) {}

  bool operator()(const double* intrinsics, double* residual) const;

  static ceres::CostFunction* Create(double target_focal, double weight);

 private:
  double target_focal_ = 0.0;
  double weight_ = 0.0;
};

// ─── Stereo intrinsics consistency factor ───────────────────────────────────
//
// Penalises left/right fx/fy/cx/cy disagreement. KITTI raw 00/01 rectified
// images should share a common rectified calibration, so this reduces
// left/right intrinsics splitting when the scene geometry is weak.
// Optimises: intrinsics_left [9], intrinsics_right [9]
// Residual dim: 4

class StereoIntrinsicsConsistencyFactor {
 public:
  explicit StereoIntrinsicsConsistencyFactor(double weight) : weight_(weight) {}

  bool operator()(const double* intrinsics_left,
                  const double* intrinsics_right,
                  double* residual) const;

  static ceres::CostFunction* Create(double weight);

 private:
  double weight_ = 0.0;
};

// ─── Principal point mean prior factor ──────────────────────────────────────
//
// Penalises cx/cy drift away from the left/right initial principal-point mean.
// This uses only the provided initial calibration, not ground truth.
// Optimises: intrinsics [9]
// Residual dim: 2

class PrincipalPointMeanPriorFactor {
 public:
  PrincipalPointMeanPriorFactor(double target_cx, double target_cy, double weight)
      : target_cx_(target_cx), target_cy_(target_cy), weight_(weight) {}

  bool operator()(const double* intrinsics, double* residual) const;

  static ceres::CostFunction* Create(double target_cx, double target_cy, double weight);

 private:
  double target_cx_ = 0.0;
  double target_cy_ = 0.0;
  double weight_ = 0.0;
};

// ─── Frame distance prior factor ───────────────────────────────────────────
//
// Penalises deviations of the distance between two optimized camera centers
// from an externally supplied metric distance. The target can come from KITTI
// OXTS odometry and is invariant to the BA world's global rotation/translation.
// Optimises: frame_a_rvec [3], frame_a_tvec [3], frame_b_rvec [3], frame_b_tvec [3]
// Residual dim: 1

class FrameDistancePriorFactor {
 public:
  FrameDistancePriorFactor(double target_distance, double weight)
      : target_distance_(target_distance), weight_(weight) {}

  bool operator()(const double* frame_a_rvec,
                  const double* frame_a_tvec,
                  const double* frame_b_rvec,
                  const double* frame_b_tvec,
                  double* residual) const;

  static ceres::CostFunction* Create(double target_distance, double weight);

 private:
  double target_distance_ = 0.0;
  double weight_ = 0.0;
};

// ─── Frame position prior factor ───────────────────────────────────────────
//
// Penalises deviations of an optimized camera center from an externally
// supplied target center in the current BA world frame. The target can be an
// OXTS trajectory aligned to the current BA trajectory by a similarity
// transform, so this constrains trajectory shape without using camera
// calibration ground truth.
// Optimises: frame_rvec [3], frame_tvec [3]
// Residual dim: 3

class FramePositionPriorFactor {
 public:
  FramePositionPriorFactor(const cv::Vec3d& target_center, double weight)
      : target_center_(target_center), weight_(weight) {}

  bool operator()(const double* frame_rvec,
                  const double* frame_tvec,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Vec3d& target_center,
                                     double weight);

 private:
  cv::Vec3d target_center_ = cv::Vec3d(0.0, 0.0, 0.0);
  double weight_ = 0.0;
};

// ─── Frame translation vector prior factor ─────────────────────────────────
//
// Penalises deviations of the relative translation from frame A to frame B,
// expressed in frame A's camera coordinates, from an externally supplied target.
// This constrains metric motion direction and scale without using camera
// intrinsics or stereo calibration ground truth.
// Optimises: frame_a_rvec [3], frame_a_tvec [3], frame_b_rvec [3], frame_b_tvec [3]
// Residual dim: 3

class FrameTranslationVectorPriorFactor {
 public:
  FrameTranslationVectorPriorFactor(const cv::Vec3d& target_delta, double weight)
      : target_delta_(target_delta), weight_(weight) {}

  bool operator()(const double* frame_a_rvec,
                  const double* frame_a_tvec,
                  const double* frame_b_rvec,
                  const double* frame_b_tvec,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Vec3d& target_delta,
                                     double weight);

 private:
  cv::Vec3d target_delta_ = cv::Vec3d(0.0, 0.0, 0.0);
  double weight_ = 0.0;
};

// ─── Frame translation direction prior factor ──────────────────────────────
//
// Penalises deviations of the unit relative translation direction from frame A
// to frame B, expressed in frame A's camera coordinates. Unlike the vector
// prior, this constrains direction without constraining metric scale.
// Optimises: frame_a_rvec [3], frame_a_tvec [3], frame_b_rvec [3], frame_b_tvec [3]
// Residual dim: 3

class FrameTranslationDirectionPriorFactor {
 public:
  FrameTranslationDirectionPriorFactor(const cv::Vec3d& target_direction,
                                       double weight)
      : target_direction_(target_direction), weight_(weight) {}

  bool operator()(const double* frame_a_rvec,
                  const double* frame_a_tvec,
                  const double* frame_b_rvec,
                  const double* frame_b_tvec,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Vec3d& target_direction,
                                     double weight);

 private:
  cv::Vec3d target_direction_ = cv::Vec3d(0.0, 0.0, 0.0);
  double weight_ = 0.0;
};

// ─── Frame rotation angle prior factor ─────────────────────────────────────
//
// Penalises deviations of the relative rotation magnitude between two
// optimized frame poses from an externally supplied relative rotation angle.
// The target can come from KITTI OXTS roll/pitch/yaw. Using only the angle
// keeps the constraint invariant to an unknown fixed camera-to-vehicle
// rotation.
// Optimises: frame_a_rvec [3], frame_b_rvec [3]
// Residual dim: 1

class FrameRotationAnglePriorFactor {
 public:
  FrameRotationAnglePriorFactor(double target_angle_rad, double weight)
      : target_angle_rad_(target_angle_rad), weight_(weight) {}

  bool operator()(const double* frame_a_rvec,
                  const double* frame_b_rvec,
                  double* residual) const;

  static ceres::CostFunction* Create(double target_angle_rad, double weight);

 private:
  double target_angle_rad_ = 0.0;
  double weight_ = 0.0;
};

// ─── Frame rotation vector prior factor ────────────────────────────────────
//
// Penalises deviations of the full relative rotation vector between two
// optimized frame poses from an externally supplied target. The target can come
// from OXTS relative roll/pitch/yaw after aligning its fixed sensor frame to
// the current BA gauge, so this constrains rotation axes without using stereo
// camera intrinsics as ground truth.
// Optimises: frame_a_rvec [3], frame_b_rvec [3]
// Residual dim: 3

class FrameRotationVectorPriorFactor {
 public:
  FrameRotationVectorPriorFactor(const cv::Vec3d& target_rvec, double weight)
      : target_rvec_(target_rvec), weight_(weight) {}

  bool operator()(const double* frame_a_rvec,
                  const double* frame_b_rvec,
                  double* residual) const;

  static ceres::CostFunction* Create(const cv::Vec3d& target_rvec,
                                     double weight);

 private:
  cv::Vec3d target_rvec_ = cv::Vec3d(0.0, 0.0, 0.0);
  double weight_ = 0.0;
};

// ─── Frame absolute rotation prior factor ──────────────────────────────────
//
// Penalises deviations of one optimized frame rotation from an externally
// supplied target rotation. The target may be aligned to the current BA gauge
// before constructing the factor, so this can constrain low-motion sequences
// without using camera intrinsics or stereo calibration ground truth.
// Optimises: frame_rvec [3]
// Residual dim: 3

class FrameAbsoluteRotationPriorFactor {
 public:
  FrameAbsoluteRotationPriorFactor(const cv::Vec3d& target_rvec, double weight)
      : target_rvec_(target_rvec), weight_(weight) {}

  bool operator()(const double* frame_rvec, double* residual) const;

  static ceres::CostFunction* Create(const cv::Vec3d& target_rvec,
                                     double weight);

 private:
  cv::Vec3d target_rvec_ = cv::Vec3d(0.0, 0.0, 0.0);
  double weight_ = 0.0;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_FACTORS_H
