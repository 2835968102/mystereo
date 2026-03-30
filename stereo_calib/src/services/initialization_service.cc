#include "services/initialization_service.h"

#include "track_builder.h"

namespace stereocalib {

FrameInitResult InitializationService::InitializeFrameRotations(
    const StereoCamera& init_camera,
    const std::vector<Track>& tracks,
    std::vector<FrameState>& frames) {
  FrameInitResult result;
  
  // Delegate to the existing function in track_builder.cc
  result.success = stereocalib::InitializeFrameRotations(
      init_camera,
      tracks,
      frames,
      result.registration_order,
      result.fixed_frame_idx);
  
  return result;
}

PointInitResult InitializationService::InitializeTrackPoints(
    const StereoCamera& init_camera,
    const std::vector<double>& extrinsics,
    const std::vector<FrameState>& frames,
    std::vector<Track>& tracks) {
  PointInitResult result;
  
  // Count initialized points before (check if point3d has valid data)
  int count_before = 0;
  for (const auto& track : tracks) {
    if (track.point3d.size() == 3 && 
        (track.point3d[0] != 0.0 || track.point3d[1] != 0.0 || track.point3d[2] != 0.0)) {
      count_before++;
    }
  }
  
  // Delegate to the existing function in track_builder.cc
  result.success = stereocalib::InitializeTrackPoints(
      init_camera,
      extrinsics,
      frames,
      tracks);
  
  // Count initialized points after
  int count_after = 0;
  for (const auto& track : tracks) {
    if (track.point3d.size() == 3 && 
        (track.point3d[0] != 0.0 || track.point3d[1] != 0.0 || track.point3d[2] != 0.0)) {
      count_after++;
    }
  }
  
  result.num_initialized = count_after - count_before;
  return result;
}

}  // namespace stereocalib
