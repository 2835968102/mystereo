#ifndef STEREO_CALIB_SRC_SERVICES_BUNDLE_ADJUSTMENT_SERVICE_H
#define STEREO_CALIB_SRC_SERVICES_BUNDLE_ADJUSTMENT_SERVICE_H

#include <ceres/ceres.h>
#include <memory>
#include <vector>

#include "stereo_types.h"
#include "track_builder.h"

namespace stereocalib {

// ─── BA Configuration ───────────────────────────────────────────────────────

struct BAConfig {
  int max_iterations = 200;
  double huber_delta = 1.0;
  bool fix_distortion = true;
  double aspect_ratio_prior_weight = 100.0;
  double baseline_prior_weight = 10.0;
};

// ─── BA State (mutable optimization state) ──────────────────────────────────

struct BAState {
  // Intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
  std::vector<double> intrinsics_left;
  std::vector<double> intrinsics_right;
  
  // Extrinsics: [r1, r2, r3, t1, t2, t3]
  std::vector<double> extrinsics;
  std::vector<double> init_extrinsics;  // For baseline prior
  
  // Frame poses and tracks
  std::vector<FrameState>* frames = nullptr;
  std::vector<Track>* tracks = nullptr;
  
  // Fixed frame index (not optimized)
  int fixed_frame_idx = 0;
};

// ─── BA Result ──────────────────────────────────────────────────────────────

struct BAResult {
  bool success = false;
  double init_rmse = 0.0;
  double final_rmse = 0.0;
  int num_residuals = 0;
  ceres::Solver::Summary summary;
};

// ─── Bundle Adjustment Service Interface ────────────────────────────────────

class IBundleAdjustmentService {
 public:
  virtual ~IBundleAdjustmentService() = default;

  /// Run bundle adjustment optimization.
  /// @param state BA state containing parameters to optimize.
  /// @param active_frames Which frames are active in this optimization.
  /// @param config BA configuration.
  /// @param frame_to_optimize Optional: only optimize this frame's pose (-1 = all).
  /// @return BA result with RMSE and summary.
  virtual BAResult RunBundleAdjustment(
      BAState& state,
      const std::vector<char>& active_frames,
      const BAConfig& config,
      int frame_to_optimize = -1) = 0;
};

// ─── Bundle Adjustment Service Implementation ───────────────────────────────

class BundleAdjustmentService : public IBundleAdjustmentService {
 public:
  BundleAdjustmentService() = default;
  ~BundleAdjustmentService() override = default;

  BAResult RunBundleAdjustment(
      BAState& state,
      const std::vector<char>& active_frames,
      const BAConfig& config,
      int frame_to_optimize = -1) override;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_SERVICES_BUNDLE_ADJUSTMENT_SERVICE_H
