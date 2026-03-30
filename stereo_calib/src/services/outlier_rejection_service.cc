#include "services/outlier_rejection_service.h"

#include <iostream>
#include <opencv2/calib3d.hpp>

#include "core/camera_math.h"
#include "stereo_factors.h"

namespace stereocalib {

int OutlierRejectionService::RejectOutliers(OutlierRejectionState& state,
                                            double threshold) {
  if (!state.frames || !state.tracks) {
    return 0;
  }

  const double thresh2 = threshold * threshold;
  int rejected_count = 0;

  std::vector<FrameState>& frames = *state.frames;
  std::vector<Track>& tracks = *state.tracks;

  // Build rotation matrix from extrinsics
  const cv::Mat rvec_rl = (cv::Mat_<double>(3, 1) 
      << state.extrinsics[0], state.extrinsics[1], state.extrinsics[2]);
  cv::Mat R_rl;
  cv::Rodrigues(rvec_rl, R_rl);
  const cv::Mat t_rl = (cv::Mat_<double>(3, 1) 
      << state.extrinsics[3], state.extrinsics[4], state.extrinsics[5]);

  for (size_t ti = 0; ti < tracks.size(); ++ti) {
    Track& track = tracks[ti];
    const cv::Mat X_w = (cv::Mat_<double>(3, 1) 
        << track.point3d[0], track.point3d[1], track.point3d[2]);

    for (TrackObservation& obs : track.observations) {
      if (obs.rejected) continue;
      if (obs.frame_idx < 0 || obs.frame_idx >= static_cast<int>(frames.size())) {
        continue;
      }

      const cv::Mat X_l = camera_math::ToRotation(frames[obs.frame_idx].rvec) * X_w;

      cv::Mat X_cam = X_l;
      const double* intr = state.intrinsics_left.data();
      if (!obs.is_left) {
        X_cam = R_rl * X_l + t_rl;
        intr = state.intrinsics_right.data();
      }

      const double Z = X_cam.at<double>(2, 0);
      if (Z <= 0.0) {
        obs.rejected = true;
        ++rejected_count;
        continue;
      }

      double u, v;
      ApplyDistAndProject(intr, X_cam.at<double>(0, 0) / Z, 
                          X_cam.at<double>(1, 0) / Z, u, v);

      const double du = obs.px.x - u;
      const double dv = obs.px.y - v;
      if (du * du + dv * dv > thresh2) {
        obs.rejected = true;
        ++rejected_count;
      }
    }
  }

  std::cout << "[Outlier Rejection] Rejected " << rejected_count
            << " observations above " << threshold << " px threshold" << std::endl;
  return rejected_count;
}

OutlierRejectionResult OutlierRejectionService::RejectOutliersIterative(
    OutlierRejectionState& state,
    const OutlierRejectionConfig& config) {
  OutlierRejectionResult result;
  
  for (int round = 0; round < config.max_rounds; ++round) {
    int rejected = RejectOutliers(state, config.threshold);
    result.rejected_count += rejected;
    result.total_rounds = round + 1;
    
    if (rejected == 0) {
      break;
    }
  }
  
  return result;
}

}  // namespace stereocalib
