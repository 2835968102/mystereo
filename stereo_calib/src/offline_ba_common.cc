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

  if (input_json.contains("left") &&
      input_json.contains("right") &&
      input_json.contains("extrinsics")) {
    try {
      gt.camera.left = IntrinsicsFromJson(input_json.at("left"));
      gt.camera.right = IntrinsicsFromJson(input_json.at("right"));
      gt.camera.extrinsics = ExtrinsicsFromJson(input_json.at("extrinsics"));
    } catch (...) {
      err = "Failed to parse ground truth camera fields from input json.";
      return false;
    }
    gt.has_gt = true;
    gt.source = "input_json(left/right/extrinsics)";
  }

  return true;
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

