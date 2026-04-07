#include "frame_score_utils.h"

#include <algorithm>

#include <opencv2/calib3d.hpp>

namespace stereocalib {
namespace {

bool IsBetterPair(const PairQuality& lhs, const PairQuality& rhs) {
  return lhs.quality_sum > rhs.quality_sum ||
         (lhs.quality_sum == rhs.quality_sum && lhs.inlier_count > rhs.inlier_count) ||
         (lhs.quality_sum == rhs.quality_sum && lhs.inlier_count == rhs.inlier_count &&
          (lhs.frame_a < rhs.frame_a ||
           (lhs.frame_a == rhs.frame_a && lhs.frame_b < rhs.frame_b)));
}

}  // namespace

bool EvaluatePairQuality(const RawImagePair& pair,
                         const std::vector<ImageInfo>& images,
                         double max_match_score,
                         int min_pair_inliers,
                         double min_pair_inlier_ratio,
                         PairQuality& quality) {
  const double kFmRansacThreshold = 2.0;
  const double kFmRansacConfidence = 0.995;
  const int kMinMatchesForRansac = 8;

  quality = PairQuality();

  const std::vector<ImageInfo>::const_iterator ita = std::find_if(
      images.begin(), images.end(), [&](const ImageInfo& info) {
        return info.name == pair.image_a;
      });
  const std::vector<ImageInfo>::const_iterator itb = std::find_if(
      images.begin(), images.end(), [&](const ImageInfo& info) {
        return info.name == pair.image_b;
      });

  if (ita == images.end() || itb == images.end()) {
    return false;
  }

  const ImageInfo& image_a = *ita;
  const ImageInfo& image_b = *itb;
  if (!image_a.valid || !image_b.valid || !image_a.is_left || !image_b.is_left) {
    return false;
  }
  if (image_a.frame_idx < 0 || image_b.frame_idx < 0 || image_a.frame_idx == image_b.frame_idx) {
    return false;
  }

  std::vector<cv::Point2f> pts_a;
  std::vector<cv::Point2f> pts_b;
  std::vector<int> idx_map;
  pts_a.reserve(pair.matches.size());
  pts_b.reserve(pair.matches.size());
  idx_map.reserve(pair.matches.size());

  for (size_t k = 0; k < pair.matches.size(); ++k) {
    const RawPairMatch& m = pair.matches[k];
    if (m.score > max_match_score) {
      continue;
    }
    pts_a.push_back(m.pt_a);
    pts_b.push_back(m.pt_b);
    idx_map.push_back(static_cast<int>(k));
  }

  if (pts_a.size() < static_cast<size_t>(kMinMatchesForRansac)) {
    return false;
  }

  cv::Mat inliers;
  const cv::Mat F = cv::findFundamentalMat(pts_a, pts_b, cv::FM_RANSAC,
                                           kFmRansacThreshold, kFmRansacConfidence, inliers);
  if (F.empty()) {
    return false;
  }

  const bool mask_row = (inliers.rows == static_cast<int>(pts_a.size()) && inliers.cols == 1);
  const bool mask_col = (inliers.cols == static_cast<int>(pts_a.size()) && inliers.rows == 1);
  if (!mask_row && !mask_col) {
    return false;
  }

  int inlier_count = 0;
  double quality_sum = 0.0;
  for (int k = 0; k < static_cast<int>(pts_a.size()); ++k) {
    const uchar ok = mask_row ? inliers.at<uchar>(k, 0) : inliers.at<uchar>(0, k);
    if (ok == 0) {
      continue;
    }
    const RawPairMatch& m = pair.matches[idx_map[k]];
    ++inlier_count;
    quality_sum += (max_match_score - m.score);
  }

  const double inlier_ratio = static_cast<double>(inlier_count) / static_cast<double>(pts_a.size());
  if (inlier_count < min_pair_inliers || inlier_ratio < min_pair_inlier_ratio) {
    return false;
  }

  quality.frame_a = image_a.frame_idx;
  quality.frame_b = image_b.frame_idx;
  quality.inlier_count = inlier_count;
  quality.quality_sum = quality_sum;
  return true;
}

bool SelectBootstrapStartFrame(const std::vector<RawImagePair>& pairs,
                               const std::vector<ImageInfo>& images,
                               double max_match_score,
                               int min_pair_inliers,
                               double min_pair_inlier_ratio,
                               int& start_frame) {
  bool found = false;
  PairQuality best_quality;

  for (size_t i = 0; i < pairs.size(); ++i) {
    PairQuality quality;
    if (!EvaluatePairQuality(pairs[i], images, max_match_score,
                             min_pair_inliers, min_pair_inlier_ratio, quality)) {
      continue;
    }

    if (!found || IsBetterPair(quality, best_quality)) {
      best_quality = quality;
      found = true;
    }
  }

  if (!found) {
    return false;
  }

  start_frame = best_quality.frame_a;
  return true;
}

bool SelectNextFrameFromPrevious(const std::vector<RawImagePair>& pairs,
                                 const std::vector<ImageInfo>& images,
                                 double max_match_score,
                                 int min_pair_inliers,
                                 double min_pair_inlier_ratio,
                                 int previous_frame,
                                 const std::vector<FrameState>& frames,
                                 int& next_frame) {
  bool found = false;
  PairQuality best_quality;

  for (size_t i = 0; i < pairs.size(); ++i) {
    PairQuality quality;
    if (!EvaluatePairQuality(pairs[i], images, max_match_score,
                             min_pair_inliers, min_pair_inlier_ratio, quality)) {
      continue;
    }

    int candidate_frame = -1;
    if (quality.frame_a == previous_frame) {
      candidate_frame = quality.frame_b;
    } else if (quality.frame_b == previous_frame) {
      candidate_frame = quality.frame_a;
    } else {
      continue;
    }

    if (candidate_frame < 0 || candidate_frame >= static_cast<int>(frames.size())) {
      continue;
    }
    if (frames[candidate_frame].initialized) {
      continue;
    }

    PairQuality candidate_quality = quality;
    if (candidate_quality.frame_a != previous_frame) {
      candidate_quality.frame_a = previous_frame;
      candidate_quality.frame_b = candidate_frame;
    }

    if (!found || IsBetterPair(candidate_quality, best_quality)) {
      best_quality = candidate_quality;
      found = true;
    }
  }

  if (!found) {
    return false;
  }

  next_frame = best_quality.frame_b;
  return true;
}

}  // namespace stereocalib
