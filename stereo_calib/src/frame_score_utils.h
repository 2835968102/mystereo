#ifndef STEREO_CALIB_SRC_FRAME_SCORE_UTILS_H
#define STEREO_CALIB_SRC_FRAME_SCORE_UTILS_H

#include <vector>

#include "track_builder.h"

namespace stereocalib {

struct PairQuality {
  int frame_a = -1;
  int frame_b = -1;
  int inlier_count = 0;
  double quality_sum = 0.0;
};

bool EvaluatePairQuality(const RawImagePair& pair,
                         const std::vector<ImageInfo>& images,
                         double max_match_score,
                         int min_pair_inliers,
                         double min_pair_inlier_ratio,
                         PairQuality& quality);

bool SelectBootstrapStartFrame(const std::vector<RawImagePair>& pairs,
                               const std::vector<ImageInfo>& images,
                               double max_match_score,
                               int min_pair_inliers,
                               double min_pair_inlier_ratio,
                               int& start_frame);

bool SelectNextFrameFromPrevious(const std::vector<RawImagePair>& pairs,
                                 const std::vector<ImageInfo>& images,
                                 double max_match_score,
                                 int min_pair_inliers,
                                 double min_pair_inlier_ratio,
                                 int previous_frame,
                                 const std::vector<FrameState>& frames,
                                 int& next_frame);

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_FRAME_SCORE_UTILS_H
