#ifndef STEREO_CALIB_SRC_SERVICES_TRACK_SERVICE_H
#define STEREO_CALIB_SRC_SERVICES_TRACK_SERVICE_H

#include <memory>
#include <vector>

#include "stereo_types.h"
#include "track_builder.h"

namespace stereocalib {

// ─── Track Build Configuration ──────────────────────────────────────────────

struct TrackBuildConfig {
  double max_match_score = 1.0;
  int min_pair_inliers = 12;
  double min_pair_inlier_ratio = 0.35;
  int min_track_len = 3;
};

// ─── Track Service Interface ────────────────────────────────────────────────

class ITrackService {
 public:
  virtual ~ITrackService() = default;

  /// Build feature tracks from raw image pair matches.
  /// @param pairs Input raw image pairs with matches.
  /// @param config Track building configuration.
  /// @param result Output track building result.
  /// @return true if tracks were successfully built.
  virtual bool BuildTracks(const std::vector<RawImagePair>& pairs,
                           const TrackBuildConfig& config,
                           TrackBuildResult& result) = 0;
};

// ─── Track Service Implementation ───────────────────────────────────────────

class TrackService : public ITrackService {
 public:
  TrackService() = default;
  ~TrackService() override = default;

  bool BuildTracks(const std::vector<RawImagePair>& pairs,
                   const TrackBuildConfig& config,
                   TrackBuildResult& result) override;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_SERVICES_TRACK_SERVICE_H
