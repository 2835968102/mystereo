/*
 * Ceres cost functions for stereo camera calibration.
 *
 * Two factor types:
 *   LeftReprojFactor   – residual between observed left pixel and projected 3D point
 *   RightReprojFactor  – residual between observed right pixel and projected 3D point
 *
 * Both use NumericDiffCostFunction with CENTRAL differences (same pattern as
 * krt_optimizer / ptzray_optimizer in the parent project) so that OpenCV
 * routines (Rodrigues, etc.) can be called freely inside operator().
 */

#ifndef STEREO_CALIB_SRC_STEREO_FACTORS_H
#define STEREO_CALIB_SRC_STEREO_FACTORS_H

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

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

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_FACTORS_H
