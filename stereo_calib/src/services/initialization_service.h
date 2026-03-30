#ifndef STEREO_CALIB_SRC_SERVICES_INITIALIZATION_SERVICE_H
#define STEREO_CALIB_SRC_SERVICES_INITIALIZATION_SERVICE_H

#include <memory>
#include <vector>

#include "stereo_types.h"
#include "track_builder.h"

namespace stereocalib {

// ─── Initialization Results ─────────────────────────────────────────────────

struct FrameInitResult {
  bool success = false;
  std::vector<int> registration_order;
  int fixed_frame_idx = 0;
};

struct PointInitResult {
  bool success = false;
  int num_initialized = 0;
};

// ─── Initialization Service Interface ───────────────────────────────────────

class IInitializationService {
 public:
  virtual ~IInitializationService() = default;

  /// Initialize frame rotations using pure rotation estimation.
  /// @param init_camera Initial stereo camera parameters.
  /// @param tracks Feature tracks.
  /// @param frames Frame states to initialize (modified in place).
  /// @return Initialization result with registration order.
  virtual FrameInitResult InitializeFrameRotations(
      const StereoCamera& init_camera,
      const std::vector<Track>& tracks,
      std::vector<FrameState>& frames) = 0;

  /// Initialize 3D track points via triangulation.
  /// @param init_camera Initial stereo camera parameters.
  /// @param extrinsics Current stereo extrinsics vector [r1,r2,r3,t1,t2,t3].
  /// @param frames Frame states.
  /// @param tracks Tracks to initialize points for (modified in place).
  /// @return Initialization result.
  virtual PointInitResult InitializeTrackPoints(
      const StereoCamera& init_camera,
      const std::vector<double>& extrinsics,
      const std::vector<FrameState>& frames,
      std::vector<Track>& tracks) = 0;
};

// ─── Initialization Service Implementation ──────────────────────────────────

class InitializationService : public IInitializationService {
 public:
  InitializationService() = default;
  ~InitializationService() override = default;

  FrameInitResult InitializeFrameRotations(
      const StereoCamera& init_camera,
      const std::vector<Track>& tracks,
      std::vector<FrameState>& frames) override;

  PointInitResult InitializeTrackPoints(
      const StereoCamera& init_camera,
      const std::vector<double>& extrinsics,
      const std::vector<FrameState>& frames,
      std::vector<Track>& tracks) override;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_SERVICES_INITIALIZATION_SERVICE_H
