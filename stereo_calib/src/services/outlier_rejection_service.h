#ifndef STEREO_CALIB_SRC_SERVICES_OUTLIER_REJECTION_SERVICE_H
#define STEREO_CALIB_SRC_SERVICES_OUTLIER_REJECTION_SERVICE_H

#include <memory>
#include <vector>

#include "stereo_types.h"
#include "track_builder.h"

namespace stereocalib {

// ─── Outlier Rejection Configuration ────────────────────────────────────────

struct OutlierRejectionConfig {
  double threshold = 2.0;  // pixels
  int max_rounds = 100;
};

// ─── Outlier Rejection State ────────────────────────────────────────────────

struct OutlierRejectionState {
  // Current camera parameters
  std::vector<double> intrinsics_left;
  std::vector<double> intrinsics_right;
  std::vector<double> extrinsics;
  
  // Frame poses and tracks (modified by rejection)
  std::vector<FrameState>* frames = nullptr;
  std::vector<Track>* tracks = nullptr;
};

// ─── Outlier Rejection Result ───────────────────────────────────────────────

struct OutlierRejectionResult {
  int rejected_count = 0;
  int total_rounds = 0;
};

// ─── Outlier Rejection Service Interface ────────────────────────────────────

class IOutlierRejectionService {
 public:
  virtual ~IOutlierRejectionService() = default;

  /// Reject outlier observations based on reprojection error.
  /// @param state Current optimization state.
  /// @param threshold Reprojection error threshold in pixels.
  /// @return Number of rejected observations.
  virtual int RejectOutliers(OutlierRejectionState& state, double threshold) = 0;
  
  /// Run iterative outlier rejection until convergence.
  /// @param state Current optimization state.
  /// @param config Outlier rejection configuration.
  /// @return Rejection result.
  virtual OutlierRejectionResult RejectOutliersIterative(
      OutlierRejectionState& state,
      const OutlierRejectionConfig& config) = 0;
};

// ─── Outlier Rejection Service Implementation ───────────────────────────────

class OutlierRejectionService : public IOutlierRejectionService {
 public:
  OutlierRejectionService() = default;
  ~OutlierRejectionService() override = default;

  int RejectOutliers(OutlierRejectionState& state, double threshold) override;
  
  OutlierRejectionResult RejectOutliersIterative(
      OutlierRejectionState& state,
      const OutlierRejectionConfig& config) override;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_SERVICES_OUTLIER_REJECTION_SERVICE_H
