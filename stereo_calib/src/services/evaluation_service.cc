#include "services/evaluation_service.h"

#include <iomanip>
#include <iostream>

#include <opencv2/calib3d.hpp>

#include "stereo_eval.h"
#include "stereo_io.h"

namespace stereocalib {

void EvaluationService::SetGroundTruth(const StereoCamera& gt) {
  ground_truth_ = gt;
  has_ground_truth_ = true;
}

bool EvaluationService::HasGroundTruth() const {
  return has_ground_truth_;
}

void EvaluationService::PrintCurrentVsGroundTruth(
    const std::string& stage_name,
    const StereoCamera& current) const {
  if (!has_ground_truth_) {
    return;
  }

  std::cout << "\n========== " << stage_name 
            << " - Comparison with Ground Truth ==========\n";
  std::cout << std::showpos << std::fixed << std::setprecision(6);

  std::cout << "Left camera:\n";
  std::cout << "  fx: " << current.left.fx << " (gt: " << ground_truth_.left.fx
            << ", diff: " << (current.left.fx - ground_truth_.left.fx) << ")\n";
  std::cout << "  fy: " << current.left.fy << " (gt: " << ground_truth_.left.fy
            << ", diff: " << (current.left.fy - ground_truth_.left.fy) << ")\n";
  std::cout << "  cx: " << current.left.cx << " (gt: " << ground_truth_.left.cx
            << ", diff: " << (current.left.cx - ground_truth_.left.cx) << ")\n";
  std::cout << "  cy: " << current.left.cy << " (gt: " << ground_truth_.left.cy
            << ", diff: " << (current.left.cy - ground_truth_.left.cy) << ")\n";

  std::cout << "Right camera:\n";
  std::cout << "  fx: " << current.right.fx << " (gt: " << ground_truth_.right.fx
            << ", diff: " << (current.right.fx - ground_truth_.right.fx) << ")\n";
  std::cout << "  fy: " << current.right.fy << " (gt: " << ground_truth_.right.fy
            << ", diff: " << (current.right.fy - ground_truth_.right.fy) << ")\n";
  std::cout << "  cx: " << current.right.cx << " (gt: " << ground_truth_.right.cx
            << ", diff: " << (current.right.cx - ground_truth_.right.cx) << ")\n";
  std::cout << "  cy: " << current.right.cy << " (gt: " << ground_truth_.right.cy
            << ", diff: " << (current.right.cy - ground_truth_.right.cy) << ")\n";

  const double tx_diff = current.extrinsics.t.at<double>(0, 0) - 
                         ground_truth_.extrinsics.t.at<double>(0, 0);
  const double ty_diff = current.extrinsics.t.at<double>(1, 0) - 
                         ground_truth_.extrinsics.t.at<double>(1, 0);
  const double tz_diff = current.extrinsics.t.at<double>(2, 0) - 
                         ground_truth_.extrinsics.t.at<double>(2, 0);

  std::cout << "Extrinsics:\n";
  std::cout << "  t: [" << current.extrinsics.t.at<double>(0, 0) << ", "
            << current.extrinsics.t.at<double>(1, 0) << ", "
            << current.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  gt: [" << ground_truth_.extrinsics.t.at<double>(0, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(1, 0) << ", "
            << ground_truth_.extrinsics.t.at<double>(2, 0) << "]\n";
  std::cout << "  diff: [" << tx_diff << ", " << ty_diff << ", " << tz_diff << "]\n";

  const double baseline_current = TranslationNorm(current.extrinsics.t);
  const double baseline_gt = TranslationNorm(ground_truth_.extrinsics.t);

  std::cout << "  baseline: " << baseline_current << " (gt: " << baseline_gt
            << ", diff: " << (baseline_current - baseline_gt) << ")\n";

  const double rot_error_deg = RotationErrorDeg(current.extrinsics.R, 
                                                 ground_truth_.extrinsics.R);
  std::cout << "  rotation_error: " << rot_error_deg << " degrees\n";

  std::cout << std::noshowpos;
  std::cout << "================================================================\n\n";
}

void EvaluationService::RecordOptimizationStage(
    const std::string& stage_name,
    double reproj_error,
    const StereoCamera& current) {
  using json = nlohmann::json;

  json stage_record;
  stage_record["stage"] = stage_name;
  stage_record["reproj_error"] = reproj_error;

  stage_record["camera"]["left"] = IntrinsicsToJson(current.left);
  stage_record["camera"]["right"] = IntrinsicsToJson(current.right);
  stage_record["camera"]["extrinsics"] = ExtrinsicsToJson(current.extrinsics);

  if (has_ground_truth_) {
    stage_record["diff_vs_gt"]["left"] = IntrinsicsDiffToJson(
        current.left, ground_truth_.left);
    stage_record["diff_vs_gt"]["right"] = IntrinsicsDiffToJson(
        current.right, ground_truth_.right);
    stage_record["diff_vs_gt"]["extrinsics"] = ExtrinsicsDiffToJson(
        current.extrinsics, ground_truth_.extrinsics);
  }

  optimization_history_.push_back(stage_record);
}

std::vector<nlohmann::json> EvaluationService::GetOptimizationHistory() const {
  return optimization_history_;
}

void EvaluationService::ClearHistory() {
  optimization_history_.clear();
}

}  // namespace stereocalib
