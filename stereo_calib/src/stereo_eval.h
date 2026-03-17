/*
 * Ground-truth comparison utilities and FOV sanity check.
 */

#ifndef STEREO_CALIB_SRC_STEREO_EVAL_H
#define STEREO_CALIB_SRC_STEREO_EVAL_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

double RotationErrorDeg(const cv::Mat& R_est, const cv::Mat& R_gt);
double TranslationNorm(const cv::Mat& t);

nlohmann::json IntrinsicsDiffToJson(const Intrinsics& est, const Intrinsics& gt);
nlohmann::json ExtrinsicsDiffToJson(const StereoExtrinsics& est, const StereoExtrinsics& gt);

void PrintDiffVsGT(const StereoCamera& est, const StereoCamera& gt, const std::string& source);
void PrintInitCamera(const StereoCamera& cam);

bool CheckFov(const std::vector<double>& intr, const char* name,
              double min_fov_deg = 10.0, double max_fov_deg = 160.0);

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_EVAL_H
