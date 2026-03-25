#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "offline_stereo_ba.h"
#include "stereo_eval.h"
#include "stereo_io.h"

using json = nlohmann::json;
using namespace stereocalib;

int main(int argc, char** argv)
{
  std::string input_path;
  std::string output_path;
  std::string gt_param_file;
  const std::string kForcedInitPathA = "stereo_calib/data/example_init_params.txt";
  const std::string kForcedInitPathB = "../data/example_init_params.txt";

  OfflineStereoBA::Options options;

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
      options.max_iter = std::stoi(argv[++i]);
    } else if (arg == "--incremental_max_iter" && i + 1 < argc) {
      options.incremental_max_iter = std::stoi(argv[++i]);
    } else if (arg == "--global_opt_interval" && i + 1 < argc) {
      options.global_opt_interval = std::stoi(argv[++i]);
    } else if (arg == "--min_track_len" && i + 1 < argc) {
      options.min_track_len = std::stoi(argv[++i]);
    } else if (arg == "--huber" && i + 1 < argc) {
      options.huber_delta = std::stod(argv[++i]);
    } else if (arg == "--max_score" && i + 1 < argc) {
      options.max_match_score = std::stod(argv[++i]);
    } else if (arg == "--min_pair_inliers" && i + 1 < argc) {
      options.min_pair_inliers = std::stoi(argv[++i]);
    } else if (arg == "--min_pair_inlier_ratio" && i + 1 < argc) {
      options.min_pair_inlier_ratio = std::stod(argv[++i]);
    } else if (arg == "--fix_distortion") {
      options.fix_distortion = true;
    } else if (arg == "--aspect_ratio_prior" && i + 1 < argc) {
      options.aspect_ratio_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--known_baseline" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --known_baseline is deprecated and ignored." << std::endl;
    } else if (arg == "--known_baseline_weight" && i + 1 < argc) {
      ++i;
      std::cout << "Warning: --known_baseline_weight is deprecated and ignored." << std::endl;
    } else if (arg == "--max_reproj_error" && i + 1 < argc) {
      options.max_reproj_error = std::stod(argv[++i]);
    } else if (arg == "--baseline_prior" && i + 1 < argc) {
      options.baseline_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--outlier_threshold" && i + 1 < argc) {
      options.outlier_rejection_threshold = std::stod(argv[++i]);
    } else if (arg == "--outlier_rounds" && i + 1 < argc) {
      options.max_outlier_rejection_rounds = std::stoi(argv[++i]);
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

  // ========== Run optimization ==========
  OfflineStereoBA optimizer(input, options);

  if (has_gt) {
    optimizer.SetGroundTruth(gt_camera);
  }

  if (poses_loaded) {
    optimizer.LoadFramePoses(input_poses_json);
  }

  StereoCamera result_camera;
  bool success = optimizer.Solve(result_camera);

  if (!success) {
    std::cerr << "Offline stereo BA did not pass the quality gate, writing best estimate anyway." << std::endl;
  }

  // ========== Write result ==========
  json out;
  out["left"] = IntrinsicsToJson(result_camera.left);
  out["right"] = IntrinsicsToJson(result_camera.right);
  out["extrinsics"] = ExtrinsicsToJson(result_camera.extrinsics);
  out["success"] = success;
  out["num_tracks"] = optimizer.num_tracks();
  out["num_observations"] = optimizer.num_observations();
  out["num_frames"] = optimizer.num_frames();
  out["init_reproj_error"] = optimizer.init_reproj_error();
  out["final_reproj_error"] = optimizer.final_reproj_error();

  out["optimization_history"] = optimizer.GetOptimizationHistory();

  if (has_gt) {
    out["gt_source"] = gt_source;
    out["diff_vs_gt"] = {
        {"left", IntrinsicsDiffToJson(result_camera.left, gt_camera.left)},
        {"right", IntrinsicsDiffToJson(result_camera.right, gt_camera.right)},
        {"extrinsics", ExtrinsicsDiffToJson(result_camera.extrinsics, gt_camera.extrinsics)},
    };
    PrintDiffVsGT(result_camera, gt_camera, gt_source);
  }

  std::ofstream fout(output_path.c_str());
  fout << out.dump(4) << std::endl;

  std::cout << "Result written to " << output_path << std::endl;
  return success ? 0 : 1;
}
