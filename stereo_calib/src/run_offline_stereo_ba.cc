#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "coordinators/optimization_coordinator.h"
#include "stereo_eval.h"
#include "stereo_io.h"

using json = nlohmann::json;
using namespace stereocalib;

namespace {

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

}  // namespace

int main(int argc, char** argv)
{
  std::string input_path;
  std::string output_path;
  std::string gt_param_file;
  const std::string kForcedInitPathA = "stereo_calib/data/example_init_params.txt";
  const std::string kForcedInitPathB = "../data/example_init_params.txt";

  OptimizationConfig config;

  // ========== Argument parsing ==========
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--init_param_file" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --init_param_file is deprecated and ignored. "
                << "Always using stereo_calib/example_init_params.txt" << std::endl;
    } else if (arg == "--gt_param_file" && i + 1 < argc) {
      gt_param_file = argv[++i];
    } else if (arg == "--use_input_init") {
      std::cout << "Warning: --use_input_init is deprecated and ignored. "
                << "Always using stereo_calib/example_init_params.txt" << std::endl;
    } else if (arg == "--init_width" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --init_width is deprecated and ignored." << std::endl;
    } else if (arg == "--init_height" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --init_height is deprecated and ignored." << std::endl;
    } else if (arg == "--init_focal" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --init_focal is deprecated and ignored." << std::endl;
    } else if (arg == "--init_baseline" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --init_baseline is deprecated and ignored." << std::endl;
    } else if (arg == "--max_iter" && i + 1 < argc) {
      config.max_iter = std::stoi(argv[++i]);
    } else if (arg == "--incremental_max_iter" && i + 1 < argc) {
      config.incremental_max_iter = std::stoi(argv[++i]);
    } else if (arg == "--global_opt_interval" && i + 1 < argc) {
      config.global_opt_interval = std::stoi(argv[++i]);
    } else if (arg == "--min_track_len" && i + 1 < argc) {
      config.min_track_len = std::stoi(argv[++i]);
    } else if (arg == "--huber" && i + 1 < argc) {
      config.huber_delta = std::stod(argv[++i]);
    } else if (arg == "--max_score" && i + 1 < argc) {
      config.max_match_score = std::stod(argv[++i]);
    } else if (arg == "--min_pair_inliers" && i + 1 < argc) {
      config.min_pair_inliers = std::stoi(argv[++i]);
    } else if (arg == "--min_pair_inlier_ratio" && i + 1 < argc) {
      config.min_pair_inlier_ratio = std::stod(argv[++i]);
    } else if (arg == "--fix_distortion") {
      config.fix_distortion = true;
    } else if (arg == "--aspect_ratio_prior" && i + 1 < argc) {
      config.aspect_ratio_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--known_baseline" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --known_baseline is deprecated and ignored." << std::endl;
    } else if (arg == "--known_baseline_weight" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --known_baseline_weight is deprecated and ignored." << std::endl;
    } else if (arg == "--max_reproj_error" && i + 1 < argc) {
      config.max_reproj_error = std::stod(argv[++i]);
    } else if (arg == "--baseline_prior" && i + 1 < argc) {
      config.baseline_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--outlier_threshold" && i + 1 < argc) {
      config.outlier_rejection_threshold = std::stod(argv[++i]);
    } else if (arg == "--outlier_rounds" && i + 1 < argc) {
      config.max_outlier_rejection_rounds = std::stoi(argv[++i]);
    }
  }

  if (input_path.empty() || output_path.empty()) {
    std::cerr << "Usage: run_offline_stereo_ba --input <matches.json> --output <result.json> "
              << "[--gt_param_file gt_params.{txt|json}] "
              << "[--max_iter 200] [--incremental_max_iter 20] [--global_opt_interval 5] "
              << "[--min_track_len 3] [--huber 1.0] [--max_score 1.0] "
              << "[--min_pair_inliers 12] [--min_pair_inlier_ratio 0.35] "
              << "[--fix_distortion] [--aspect_ratio_prior 1.0] "
              << "[--baseline_prior 10.0] [--max_reproj_error 20.0] "
              << "[--outlier_threshold 2.0] [--outlier_rounds 3]\n"
              << "Initial values are always loaded from stereo_calib/example_init_params.txt"
              << std::endl;
    return 1;
  }

  // ========== Load input data ==========
  std::ifstream fin(input_path.c_str());
  if (!fin.is_open()) {
    std::cerr << "Cannot open input file: " << input_path << std::endl;
    return 1;
  }

  json j;
  fin >> j;

  if (!j.contains("pairs")) {
    std::cerr << "Input json must contain: pairs" << std::endl;
    return 1;
  }

  OfflineBAInput input;
  input.pairs = RawPairsFromJson(j.at("pairs"));

  // ========== Load initial camera parameters ==========
  std::string err;
  if (!LoadCameraFromFile(kForcedInitPathA, input.init_camera, err)) {
    if (!LoadCameraFromFile(kForcedInitPathB, input.init_camera, err)) {
      std::cerr << err << std::endl;
      return 1;
    }
    std::cout << "Init mode: fixed_file (" << kForcedInitPathB << ")" << std::endl;
  } else {
    std::cout << "Init mode: fixed_file (" << kForcedInitPathA << ")" << std::endl;
  }

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Initial camera parameters used by optimizer:" << std::endl;
  PrintInitCamera(input.init_camera);

  std::size_t raw_matches = 0;
  for (std::size_t i = 0; i < input.pairs.size(); ++i) {
    raw_matches += input.pairs[i].matches.size();
  }

  std::cout << "Loaded " << input.pairs.size() << " pair records, "
            << raw_matches << " raw matches." << std::endl;

  // ========== Filter pairs with too few matches ==========
  const int kMinMatchesPerPair = 50;
  std::vector<RawImagePair> filtered_pairs;
  std::size_t filtered_matches = 0;
  std::size_t rejected_pairs = 0;

  for (std::size_t i = 0; i < input.pairs.size(); ++i) {
    if (input.pairs[i].matches.size() >= kMinMatchesPerPair) {
      filtered_pairs.push_back(input.pairs[i]);
      filtered_matches += input.pairs[i].matches.size();
    } else {
      rejected_pairs++;
    }
  }

  input.pairs = filtered_pairs;

  std::cout << "Filtered pairs: " << rejected_pairs << " pairs rejected (< "
            << kMinMatchesPerPair << " matches), "
            << input.pairs.size() << " pairs remaining with "
            << filtered_matches << " matches." << std::endl;

  // Load ground truth (optional)
  StereoCamera gt_camera;
  bool has_gt = false;
  std::string gt_source;
  if (!gt_param_file.empty()) {
    std::string gt_err;
    if (!LoadCameraFromFile(gt_param_file, gt_camera, gt_err)) {
      std::cerr << gt_err << std::endl;
      return 1;
    }
    has_gt = true;
    gt_source = std::string("gt_param_file: ") + gt_param_file;
  } else if (j.contains("left") && j.contains("right") && j.contains("extrinsics")) {
    gt_camera.left = IntrinsicsFromJson(j.at("left"));
    gt_camera.right = IntrinsicsFromJson(j.at("right"));
    gt_camera.extrinsics = ExtrinsicsFromJson(j.at("extrinsics"));
    has_gt = true;
    gt_source = "input_json(left/right/extrinsics)";
  }

  if (has_gt) {
    std::cout << "Ground truth loaded from: " << gt_source << std::endl;
  }

  // ========== Load frame poses (optional) ==========
  const std::string kForcedPosesPathA = "stereo_calib/data/camera_poses.json";
  const std::string kForcedPosesPathB = "../data/camera_poses.json";
  json input_poses_json;
  bool poses_loaded = false;

  std::ifstream poses_fin(kForcedPosesPathA.c_str());
  if (!poses_fin.is_open()) {
    poses_fin.close();
    poses_fin.open(kForcedPosesPathB.c_str());
  }

  if (poses_fin.is_open()) {
    try {
      poses_fin >> input_poses_json;
      poses_loaded = true;
      std::cout << "Loaded camera poses from camera_poses.json" << std::endl;
    } catch (...) {
      std::cerr << "Warning: Failed to parse camera_poses.json" << std::endl;
    }
    poses_fin.close();
  }

  // ========== Run optimization using OptimizationCoordinator ==========
  OptimizationCoordinator coordinator;

  if (has_gt) {
    coordinator.SetGroundTruth(gt_camera);
  }

  if (poses_loaded) {
    coordinator.LoadFramePoses(input_poses_json);
  }

  OptimizationResult result = coordinator.RunIncrementalBA(input, config);

  if (!result.success) {
    std::cerr << "Offline stereo BA did not pass the quality gate, writing best estimate anyway." << std::endl;
  }

  // ========== Write result ==========
  json out;
  out["left"] = IntrinsicsToJson(result.camera.left);
  out["right"] = IntrinsicsToJson(result.camera.right);
  out["extrinsics"] = ExtrinsicsToJson(result.camera.extrinsics);
  out["success"] = result.success;
  out["num_tracks"] = result.num_tracks;
  out["num_observations"] = result.num_observations;
  out["num_frames"] = result.num_frames;
  out["init_reproj_error"] = result.init_reproj_error;
  out["final_reproj_error"] = result.final_reproj_error;

  out["optimization_history"] = result.optimization_history;
  out["summary"] = BuildSummaryFromHistory(result.optimization_history);

  if (has_gt) {
    out["gt_source"] = gt_source;
    out["diff_vs_gt"] = {
        {"left", IntrinsicsDiffToJson(result.camera.left, gt_camera.left)},
        {"right", IntrinsicsDiffToJson(result.camera.right, gt_camera.right)},
        {"extrinsics", ExtrinsicsDiffToJson(result.camera.extrinsics, gt_camera.extrinsics)},
    };
    PrintDiffVsGT(result.camera, gt_camera, gt_source);
  }

  std::ofstream fout(output_path.c_str());
  fout << out.dump(4) << std::endl;

  std::cout << "Result written to " << output_path << std::endl;
  return result.success ? 0 : 1;
}
