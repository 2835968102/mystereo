#include "services/track_service.h"

#include "track_builder.h"

namespace stereocalib {

bool TrackService::BuildTracks(const std::vector<RawImagePair>& pairs,
                               const TrackBuildConfig& config,
                               TrackBuildResult& result) {
  // Delegate to the existing BuildTracks function in track_builder.cc
  return stereocalib::BuildTracks(pairs,
                                  config.max_match_score,
                                  config.min_pair_inliers,
                                  config.min_pair_inlier_ratio,
                                  config.min_track_len,
                                  result);
}

}  // namespace stereocalib
