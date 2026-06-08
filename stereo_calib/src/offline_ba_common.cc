#include "offline_ba_common.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "stereo_eval.h"
#include "stereo_io.h"

namespace stereocalib {
namespace {

using json = nlohmann::json;

double JsonAbsOrNaN(const json& obj, const char* key)
{
  if (!obj.contains(key) || !obj.at(key).is_number()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::abs(obj.at(key).get<double>());
}

double JsonIndexAbsOrNaN(const json& arr, std::size_t index)
{
  if (!arr.is_array() || index >= arr.size() || !arr.at(index).is_number()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::abs(arr.at(index).get<double>());
}

void AddIfFinite(double value, double& sum, int& count)
{
  if (std::isfinite(value)) {
    sum += value;
    ++count;
  }
}

json MeanOrNull(double sum, int count)
{
  if (count == 0) {
    return nullptr;
  }
  return sum / static_cast<double>(count);
}

double MaxAbsIntrinsicDelta(const Intrinsics& a,
                            const Intrinsics& b,
                            bool include_principal_point)
{
  double max_delta = std::max(std::abs(a.fx - b.fx), std::abs(a.fy - b.fy));
  if (include_principal_point) {
    max_delta = std::max(max_delta, std::abs(a.cx - b.cx));
    max_delta = std::max(max_delta, std::abs(a.cy - b.cy));
  }
  return max_delta;
}

double MaxAbsStereoIntrinsicDelta(const StereoCamera& a,
                                  const StereoCamera& b,
                                  bool include_principal_point)
{
  return std::max(
      MaxAbsIntrinsicDelta(a.left, b.left, include_principal_point),
      MaxAbsIntrinsicDelta(a.right, b.right, include_principal_point));
}

double FocalMean(const StereoCamera& camera)
{
  return 0.25 * (camera.left.fx + camera.left.fy +
                 camera.right.fx + camera.right.fy);
}

double TranslationNorm(const StereoCamera& camera)
{
  if (camera.extrinsics.t.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double tx = camera.extrinsics.t.at<double>(0, 0);
  const double ty = camera.extrinsics.t.at<double>(1, 0);
  const double tz = camera.extrinsics.t.at<double>(2, 0);
  return std::sqrt(tx * tx + ty * ty + tz * tz);
}

double Tx(const StereoCamera& camera)
{
  if (camera.extrinsics.t.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return camera.extrinsics.t.at<double>(0, 0);
}

}  // namespace

bool LoadJsonFile(const std::string& path, json& parsed)
{
  std::ifstream fin(path.c_str());
  if (!fin.is_open()) {
    return false;
  }
  try {
    fin >> parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool LoadOfflineBAInputJson(const std::string& input_path,
                            json& parsed,
                            OfflineBAInput& input,
                            std::string& err)
{
  std::ifstream fin(input_path.c_str());
  if (!fin.is_open()) {
    err = "Cannot open input file: " + input_path;
    return false;
  }

  try {
    fin >> parsed;
  } catch (...) {
    err = "Invalid input json file: " + input_path;
    return false;
  }

  if (!parsed.contains("pairs")) {
    err = "Input json must contain: pairs";
    return false;
  }

  input.pairs = RawPairsFromJson(parsed.at("pairs"));
  return true;
}

std::size_t CountRawMatches(const std::vector<RawImagePair>& pairs)
{
  std::size_t raw_matches = 0;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    raw_matches += pairs[i].matches.size();
  }
  return raw_matches;
}

PairFilterStats FilterPairsByMinMatches(std::vector<RawImagePair>& pairs,
                                         int min_matches_per_pair)
{
  PairFilterStats stats;
  std::vector<RawImagePair> filtered_pairs;

  for (std::size_t i = 0; i < pairs.size(); ++i) {
    if (pairs[i].matches.size() >= static_cast<std::size_t>(min_matches_per_pair)) {
      filtered_pairs.push_back(pairs[i]);
      stats.remaining_matches += pairs[i].matches.size();
    } else {
      stats.rejected_pairs++;
    }
  }

  pairs = filtered_pairs;
  stats.remaining_pairs = pairs.size();
  return stats;
}

bool LoadGroundTruth(const std::string& gt_param_file,
                     const json& input_json,
                     GroundTruthContext& gt,
                     std::string& err)
{
  gt = GroundTruthContext();
  if (!gt_param_file.empty()) {
    if (!LoadCameraFromFile(gt_param_file, gt.camera, err)) {
      return false;
    }
    gt.has_gt = true;
    gt.source = std::string("gt_param_file: ") + gt_param_file;
    return true;
  }

  return true;
}

bool RejectGtLikeInitialization(const StereoCamera& init_camera,
                                const GroundTruthContext& gt,
                                bool normalize_initial_focal_to_mean,
                                bool fix_focal_length,
                                const std::string& init_source,
                                std::string& err)
{
  if (!gt.has_gt) {
    return true;
  }

  std::vector<std::string> reasons;
  const double kFocalEps = 1e-3;
  const double kTranslationEps = 1e-6;

  if (normalize_initial_focal_to_mean &&
      std::abs(FocalMean(init_camera) - FocalMean(gt.camera)) < kFocalEps) {
    reasons.push_back(
        "normalize_initial_focal_to_mean would set fx/fy to the GT focal mean");
  }
  if (fix_focal_length &&
      MaxAbsStereoIntrinsicDelta(init_camera, gt.camera, false) < kFocalEps) {
    reasons.push_back("fix_focal_length would lock GT-like focal values");
  }
  if (MaxAbsStereoIntrinsicDelta(init_camera, gt.camera, true) < kFocalEps &&
      std::abs(TranslationNorm(init_camera) - TranslationNorm(gt.camera)) < kTranslationEps &&
      std::abs(Tx(init_camera) - Tx(gt.camera)) < kTranslationEps) {
    reasons.push_back("the init camera is numerically identical to the GT camera");
  }

  if (reasons.empty()) {
    return true;
  }

  err = "Refusing to run BA because the initial camera appears to leak evaluation GT:\n"
        "  init source: " + init_source + "\n"
        "  gt source: " + gt.source;
  for (std::size_t i = 0; i < reasons.size(); ++i) {
    err += "\n  - " + reasons[i];
  }
  err += "\nUse a GT-free init file or disable the GT-like focal normalization/fix.";
  return false;
}

json BuildSummaryFromHistory(const json& history)
{
  double reproj_sum = 0.0;
  int reproj_count = 0;
  double rotation_sum = 0.0;
  int rotation_count = 0;
  double left_fx_sum = 0.0;
  int left_fx_count = 0;
  double left_fy_sum = 0.0;
  int left_fy_count = 0;
  double right_fx_sum = 0.0;
  int right_fx_count = 0;
  double right_fy_sum = 0.0;
  int right_fy_count = 0;
  double baseline_sum = 0.0;
  int baseline_count = 0;
  double tx_sum = 0.0;
  int tx_count = 0;
  double ty_sum = 0.0;
  int ty_count = 0;
  double tz_sum = 0.0;
  int tz_count = 0;
  double focal_sum = 0.0;
  int focal_count = 0;

  if (!history.is_array()) {
    return json::object();
  }

  for (const auto& item : history) {
    AddIfFinite(item.value("reproj_error", std::numeric_limits<double>::quiet_NaN()), reproj_sum, reproj_count);

    const json& diff_vs_gt = item.contains("diff_vs_gt") ? item.at("diff_vs_gt") : json::object();
    const json& extrinsics = diff_vs_gt.contains("extrinsics") ? diff_vs_gt.at("extrinsics") : json::object();
    const json& left = diff_vs_gt.contains("left") ? diff_vs_gt.at("left") : json::object();
    const json& right = diff_vs_gt.contains("right") ? diff_vs_gt.at("right") : json::object();

    AddIfFinite(JsonAbsOrNaN(extrinsics, "rotation_error_deg"), rotation_sum, rotation_count);
    AddIfFinite(JsonAbsOrNaN(extrinsics, "baseline"), baseline_sum, baseline_count);

    const json& t = extrinsics.contains("t") ? extrinsics.at("t") : json::array();
    AddIfFinite(JsonIndexAbsOrNaN(t, 0), tx_sum, tx_count);
    AddIfFinite(JsonIndexAbsOrNaN(t, 1), ty_sum, ty_count);
    AddIfFinite(JsonIndexAbsOrNaN(t, 2), tz_sum, tz_count);

    const double left_fx = JsonAbsOrNaN(left, "fx");
    const double left_fy = JsonAbsOrNaN(left, "fy");
    const double right_fx = JsonAbsOrNaN(right, "fx");
    const double right_fy = JsonAbsOrNaN(right, "fy");

    AddIfFinite(left_fx, left_fx_sum, left_fx_count);
    AddIfFinite(left_fy, left_fy_sum, left_fy_count);
    AddIfFinite(right_fx, right_fx_sum, right_fx_count);
    AddIfFinite(right_fy, right_fy_sum, right_fy_count);

    AddIfFinite(left_fx, focal_sum, focal_count);
    AddIfFinite(left_fy, focal_sum, focal_count);
    AddIfFinite(right_fx, focal_sum, focal_count);
    AddIfFinite(right_fy, focal_sum, focal_count);
  }

  return {
      {"avg_reproj_error_px", MeanOrNull(reproj_sum, reproj_count)},
      {"avg_rotation_error_deg", MeanOrNull(rotation_sum, rotation_count)},
      {"avg_left_fx_error_px", MeanOrNull(left_fx_sum, left_fx_count)},
      {"avg_left_fy_error_px", MeanOrNull(left_fy_sum, left_fy_count)},
      {"avg_right_fx_error_px", MeanOrNull(right_fx_sum, right_fx_count)},
      {"avg_right_fy_error_px", MeanOrNull(right_fy_sum, right_fy_count)},
      {"avg_baseline_error_m", MeanOrNull(baseline_sum, baseline_count)},
      {"avg_trans_err_x_m", MeanOrNull(tx_sum, tx_count)},
      {"avg_trans_err_y_m", MeanOrNull(ty_sum, ty_count)},
      {"avg_trans_err_z_m", MeanOrNull(tz_sum, tz_count)},
      {"avg_focal_error_px", MeanOrNull(focal_sum, focal_count)},
  };
}

json BuildResultJson(const OptimizationResult& result,
                     bool include_conflict_stats)
{
  json out;
  out["left"] = IntrinsicsToJson(result.camera.left);
  out["right"] = IntrinsicsToJson(result.camera.right);
  out["extrinsics"] = ExtrinsicsToJson(result.camera.extrinsics);
  out["success"] = result.success;
  out["num_tracks"] = result.num_tracks;
  out["num_observations"] = result.num_observations;
  out["num_frames"] = result.num_frames;
  if (include_conflict_stats) {
    out["num_conflicted_components"] = result.num_conflicted_components;
    out["num_conflict_observations_skipped"] = result.num_conflict_observations_skipped;
    out["num_components_skipped_due_to_conflict"] = result.num_components_skipped_due_to_conflict;
  }
  out["init_reproj_error"] = result.init_reproj_error;
  out["final_reproj_error"] = result.final_reproj_error;
  out["optimization_history"] = result.optimization_history;
  out["outlier_rejection_history"] = result.outlier_rejection_history;
  out["summary"] = BuildSummaryFromHistory(result.optimization_history);
  return out;
}

void AppendGroundTruthDiff(json& out,
                           const OptimizationResult& result,
                           const GroundTruthContext& gt)
{
  if (!gt.has_gt) {
    return;
  }
  out["gt_source"] = gt.source;
  out["diff_vs_gt"] = {
      {"left", IntrinsicsDiffToJson(result.camera.left, gt.camera.left)},
      {"right", IntrinsicsDiffToJson(result.camera.right, gt.camera.right)},
      {"extrinsics", ExtrinsicsDiffToJson(result.camera.extrinsics, gt.camera.extrinsics)},
  };
  PrintDiffVsGT(result.camera, gt.camera, gt.source);
}

}  // namespace stereocalib
