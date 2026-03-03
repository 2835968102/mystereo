/*
 * StereoOptimizer – bundle adjustment for stereo camera calibration.
 *
 * Jointly optimises:
 *   • Left  camera intrinsics  [fx,fy,cx,cy,k1,k2,p1,p2,k3]
 *   • Right camera intrinsics  [fx,fy,cx,cy,k1,k2,p1,p2,k3]
 *   • Stereo extrinsics        [r1,r2,r3,t1,t2,t3]  (R_rl, t_rl)
 *   • 3D points                [X,Y,Z] in the left camera frame (one per match)
 *
 * Algorithm:
 *   1. Triangulate initial 3D points from stereo correspondences.
 *   2. Build a Ceres problem with LeftReprojFactor + RightReprojFactor for
 *      every correspondence.
 *   3. Solve with SPARSE_SCHUR (standard choice for bundle adjustment).
 *   4. Extract refined camera parameters.
 */

#ifndef STEREO_CALIB_SRC_STEREO_OPTIMIZER_H
#define STEREO_CALIB_SRC_STEREO_OPTIMIZER_H

#include <ceres/ceres.h>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

class StereoOptimizer {
 public:
  /**
   * @param pairs        Stereo match pairs (input data)
   * @param init_camera  Initial camera parameters (starting point for optimisation)
   * @param max_iter     Maximum number of Ceres iterations
   * @param max_reproj_error  Accept threshold for final reprojection error (pixels)
   */
  StereoOptimizer(const std::vector<StereoPair>& pairs, const StereoCamera& init_camera, int max_iter,
                  double max_reproj_error = 3.0);
  ~StereoOptimizer() = default;

  /**
   * Run the optimisation.
   * @param camera  Output: refined camera parameters.
   * @return true if Ceres converged and error is within threshold.
   */
  bool Solve(StereoCamera& camera);

  double init_reproj_error()  const { return init_reproj_error_; }
  double final_reproj_error() const { return final_reproj_error_; }

 private:
  void Triangulate(const StereoCamera& camera);
  void SetUpParams(const StereoCamera& camera);
  void AddReprojConstraints();
  void ObtainResults(StereoCamera& camera);
  void CalReprojError();
  bool CheckResults() const;

 private:
  std::vector<StereoPair> pairs_;
  int    max_iter_;
  double max_reproj_error_;

  // ── Ceres parameter blocks ────────────────────────────────────────────────
  std::vector<double> intrinsics_left_;   // [fx,fy,cx,cy,k1,k2,p1,p2,k3]
  std::vector<double> intrinsics_right_;  // [fx,fy,cx,cy,k1,k2,p1,p2,k3]
  std::vector<double> extrinsics_;        // [r1,r2,r3,t1,t2,t3]

  // points3d_[pair_idx][match_idx] = {X, Y, Z} in left camera frame
  std::vector<std::vector<std::vector<double>>> points3d_;

  ceres::Problem          problem_;
  ceres::Solver::Summary  summary_;

  double init_reproj_error_  = 0.0;
  double final_reproj_error_ = 0.0;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_OPTIMIZER_H
