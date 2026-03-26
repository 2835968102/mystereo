/*
 * Geometric utility functions for camera calibration.
 *
 * Common math operations used across multiple modules.
 */

#ifndef STEREO_CALIB_SRC_CORE_CAMERA_MATH_H
#define STEREO_CALIB_SRC_CORE_CAMERA_MATH_H

#include <opencv2/core.hpp>
#include <vector>

namespace stereocalib {
namespace camera_math {

// ─── Rotation conversions ───────────────────────────────────────────────────

// Convert Rodrigues vector (as std::vector<double>) to rotation matrix
cv::Mat ToRotation(const std::vector<double>& rvec);

// Convert rotation matrix to Rodrigues vector (as std::vector<double>)
std::vector<double> ToRodrigues(const cv::Mat& R);

// ─── 3D geometry ─────────────────────────────────────────────────────────────

// Triangulate a 3D point from stereo correspondences
// Returns true if successful, false if triangulation fails
bool TriangulatePoint(const cv::Point2f& pt_left,
                      const cv::Point2f& pt_right,
                      const cv::Mat& K_left,
                      const cv::Mat& K_right,
                      const cv::Mat& dist_left,
                      const cv::Mat& dist_right,
                      const cv::Mat& R_rl,
                      const cv::Mat& t_rl,
                      cv::Point3d& point_3d);

}  // namespace camera_math
}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_CORE_CAMERA_MATH_H
