#include "stereo_eval.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace stereocalib {

double RotationErrorDeg(const cv::Mat& R_est, const cv::Mat& R_gt)
{
  static const double kRad2Deg = 57.2957795130823208768;
  cv::Mat R_diff = R_est * R_gt.t();
  const double tr = R_diff.at<double>(0, 0) + R_diff.at<double>(1, 1) + R_diff.at<double>(2, 2);
  double cos_theta = (tr - 1.0) * 0.5;
  cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
  return std::acos(cos_theta) * kRad2Deg;
}

double TranslationNorm(const cv::Mat& t)
{
  return std::sqrt(t.at<double>(0, 0) * t.at<double>(0, 0) +
                   t.at<double>(1, 0) * t.at<double>(1, 0) +
                   t.at<double>(2, 0) * t.at<double>(2, 0));
}

nlohmann::json IntrinsicsDiffToJson(const Intrinsics& est, const Intrinsics& gt)
{
  return {
      {"fx", est.fx - gt.fx},
      {"fy", est.fy - gt.fy},
      {"cx", est.cx - gt.cx},
      {"cy", est.cy - gt.cy},
      {"k1", est.k1 - gt.k1},
      {"k2", est.k2 - gt.k2},
      {"p1", est.p1 - gt.p1},
      {"p2", est.p2 - gt.p2},
      {"k3", est.k3 - gt.k3},
  };
}

nlohmann::json ExtrinsicsDiffToJson(const StereoExtrinsics& est, const StereoExtrinsics& gt)
{
  std::vector<double> dR(9, 0.0);
  std::vector<double> dt(3, 0.0);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      dR[r * 3 + c] = est.R.at<double>(r, c) - gt.R.at<double>(r, c);
    }
  }
  for (int i = 0; i < 3; ++i) {
    dt[i] = est.t.at<double>(i, 0) - gt.t.at<double>(i, 0);
  }

  const double baseline_est = TranslationNorm(est.t);
  const double baseline_gt = TranslationNorm(gt.t);

  return {
      {"R", dR},
      {"t", dt},
      {"baseline", baseline_est - baseline_gt},
      {"rotation_error_deg", RotationErrorDeg(est.R, gt.R)},
  };
}

void PrintDiffVsGT(const StereoCamera& est, const StereoCamera& gt, const std::string& source)
{
  std::cout << "Comparison vs ground truth (" << source << "), delta = estimate - gt:" << std::endl;
  std::cout << std::showpos;
  std::cout << "  left:  fx=" << (est.left.fx - gt.left.fx)
            << ", fy=" << (est.left.fy - gt.left.fy)
            << ", cx=" << (est.left.cx - gt.left.cx)
            << ", cy=" << (est.left.cy - gt.left.cy) << std::endl;
  std::cout << "  right: fx=" << (est.right.fx - gt.right.fx)
            << ", fy=" << (est.right.fy - gt.right.fy)
            << ", cx=" << (est.right.cx - gt.right.cx)
            << ", cy=" << (est.right.cy - gt.right.cy) << std::endl;

  const double dtx = est.extrinsics.t.at<double>(0, 0) - gt.extrinsics.t.at<double>(0, 0);
  const double dty = est.extrinsics.t.at<double>(1, 0) - gt.extrinsics.t.at<double>(1, 0);
  const double dtz = est.extrinsics.t.at<double>(2, 0) - gt.extrinsics.t.at<double>(2, 0);
  const double baseline_est = TranslationNorm(est.extrinsics.t);
  const double baseline_gt = TranslationNorm(gt.extrinsics.t);
  std::cout << "  extrinsics: dt=[" << dtx << ", " << dty << ", " << dtz << "]"
            << ", d|t|=" << (baseline_est - baseline_gt)
            << ", rot_err_deg=" << RotationErrorDeg(est.extrinsics.R, gt.extrinsics.R)
            << std::endl;
  std::cout << std::noshowpos;
}

void PrintInitCamera(const StereoCamera& cam)
{
  auto print_intr = [](const std::string& name, const Intrinsics& intr) {
    std::cout << name << " intrinsics: "
              << "fx=" << intr.fx << ", fy=" << intr.fy
              << ", cx=" << intr.cx << ", cy=" << intr.cy
              << ", k1=" << intr.k1 << ", k2=" << intr.k2
              << ", p1=" << intr.p1 << ", p2=" << intr.p2
              << ", k3=" << intr.k3 << std::endl;
  };

  print_intr("left", cam.left);
  print_intr("right", cam.right);

  std::vector<double> ext = cam.extrinsics.ToVector();
  std::cout << "extrinsics (Rodrigues+t): "
            << "rvec=[" << ext[0] << ", " << ext[1] << ", " << ext[2] << "], "
            << "t=[" << ext[3] << ", " << ext[4] << ", " << ext[5] << "]"
            << std::endl;

  std::cout << "extrinsics R (row-major): [";
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (r != 0 || c != 0) {
        std::cout << ", ";
      }
      std::cout << cam.extrinsics.R.at<double>(r, c);
    }
  }
  std::cout << "]" << std::endl;
}

bool CheckFov(const std::vector<double>& intr, const char* name,
              double min_fov_deg, double max_fov_deg)
{
  const double kDegPerRad = 180.0 / M_PI;
  const double fx = intr[0];
  const double cx = intr[2];
  if (fx <= 0.0 || cx <= 0.0) {
    std::cerr << name << " has non-positive fx or cx after optimisation." << std::endl;
    return false;
  }
  const double fov_deg = 2.0 * std::atan(cx / fx) * kDegPerRad;
  if (fov_deg < min_fov_deg || fov_deg > max_fov_deg) {
    std::cerr << name << " estimated FOV " << fov_deg << " deg is outside ["
              << min_fov_deg << ", " << max_fov_deg << "] deg." << std::endl;
    return false;
  }
  return true;
}

}  // namespace stereocalib
