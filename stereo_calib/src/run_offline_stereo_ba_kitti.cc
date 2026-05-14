#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "coordinators/optimization_coordinator.h"
#include "offline_ba_common.h"
#include "stereo_eval.h"
#include "stereo_io.h"

using json = nlohmann::json;
using namespace stereocalib;

namespace {

void PrintUsage()
{
  std::cerr << "Usage: run_offline_stereo_ba_kitti --input <matches.json> --output <result.json> "
            << "[--gt_param_file gt_params.{txt|json}] "
            << "[--frame_poses_file camera_poses.json] "
            << "[--init_param_file example_init_params.txt] "
            << "[--max_iter 200] [--incremental_max_iter 20] [--per_frame_max_iter 5] [--global_opt_interval 5] "
            << "[--enable_per_frame_correction] [--min_track_len 3] [--huber 1.0] [--max_score 1.0] "
            << "[--min_pair_inliers 12] [--min_pair_inlier_ratio 0.35] "
            << "[--fix_distortion] [KITTI default: optimize principal point after a fixed-principal-point final BA pass] [--aspect_ratio_prior 1.0] "
            << "[--baseline_prior 10.0] [--tx_prior 0.0] [--focal_prior 0.0] [--focal_lower_scale 0.5] [--focal_upper_scale 1.5] "
            << "[--reset_camera_params_each_ba_round] "
            << "[--max_reproj_error 20.0] [--outlier_threshold 2.0] [--outlier_rounds 3]"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string input_path;
  std::string output_path;
  std::string gt_param_file;
  std::string frame_poses_file;
  std::string init_param_file;
  const std::string kDefaultInitPathA = "stereo_calib/data/example_init_params.txt";
  const std::string kDefaultInitPathB = "../data/example_init_params.txt";
  const std::string kDefaultPosesPathA = "stereo_calib/data/camera_poses.json";
  const std::string kDefaultPosesPathB = "../data/camera_poses.json";

  OptimizationConfig config;
  config.fix_principal_point = false;
  config.enable_two_stage_final_global_ba = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--gt_param_file" && i + 1 < argc) {
      gt_param_file = argv[++i];
    } else if (arg == "--frame_poses_file" && i + 1 < argc) {
      frame_poses_file = argv[++i];
    } else if (arg == "--init_param_file" && i + 1 < argc) {
      init_param_file = argv[++i];
    } else if (arg == "--max_iter" && i + 1 < argc) {
      config.max_iter = std::stoi(argv[++i]);
    } else if (arg == "--incremental_max_iter" && i + 1 < argc) {
      config.incremental_max_iter = std::stoi(argv[++i]);
    } else if (arg == "--per_frame_max_iter" && i + 1 < argc) {
      config.per_frame_max_iter = std::stoi(argv[++i]);
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
    } else if (arg == "--max_reproj_error" && i + 1 < argc) {
      config.max_reproj_error = std::stod(argv[++i]);
    } else if (arg == "--baseline_prior" && i + 1 < argc) {
      config.baseline_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--tx_prior" && i + 1 < argc) {
      config.tx_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--focal_prior" && i + 1 < argc) {
      config.focal_prior_weight = std::stod(argv[++i]);
    } else if (arg == "--focal_lower_scale" && i + 1 < argc) {
      config.focal_lower_scale = std::stod(argv[++i]);
    } else if (arg == "--focal_upper_scale" && i + 1 < argc) {
      config.focal_upper_scale = std::stod(argv[++i]);
    } else if (arg == "--enable_per_frame_correction") {
      config.enable_per_frame_correction = true;
    } else if (arg == "--reset_camera_params_each_ba_round") {
      config.reset_camera_params_each_ba_round = true;
    } else if (arg == "--outlier_threshold" && i + 1 < argc) {
      config.outlier_rejection_threshold = std::stod(argv[++i]);
    } else if (arg == "--outlier_rounds" && i + 1 < argc) {
      config.max_outlier_rejection_rounds = std::stoi(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      PrintUsage();
      return 1;
    }
  }

  if (input_path.empty() || output_path.empty()) {
    PrintUsage();
    return 1;
  }

  json j;
  OfflineBAInput input;
  std::string err;
  if (!LoadOfflineBAInputJson(input_path, j, input, err)) {
    std::cerr << err << std::endl;
    return 1;
  }

  bool init_loaded = false;
  if (!init_param_file.empty()) {
    init_loaded = LoadCameraFromFile(init_param_file, input.init_camera, err);
    if (!init_loaded) {
      std::cerr << err << std::endl;
      return 1;
    }
    std::cout << "Init mode: explicit_file (" << init_param_file << ")" << std::endl;
  } else if (LoadCameraFromFile(kDefaultInitPathA, input.init_camera, err)) {
    init_loaded = true;
    std::cout << "Init mode: fixed_file (" << kDefaultInitPathA << ")" << std::endl;
  } else if (LoadCameraFromFile(kDefaultInitPathB, input.init_camera, err)) {
    init_loaded = true;
    std::cout << "Init mode: fixed_file (" << kDefaultInitPathB << ")" << std::endl;
  }
  if (!init_loaded) {
    std::cerr << err << std::endl;
    return 1;
  }

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Initial camera parameters used by KITTI optimizer:" << std::endl;
  PrintInitCamera(input.init_camera);

  const std::size_t raw_matches = CountRawMatches(input.pairs);
  std::cout << "Loaded " << input.pairs.size() << " pair records, "
            << raw_matches << " raw matches." << std::endl;
  std::cout << "Experiment reset_camera_params_each_ba_round: "
            << (config.reset_camera_params_each_ba_round ? "enabled" : "disabled")
            << std::endl;

  const int kMinMatchesPerPair = 50;
  const PairFilterStats filter_stats =
      FilterPairsByMinMatches(input.pairs, kMinMatchesPerPair);
  std::cout << "Filtered pairs: " << filter_stats.rejected_pairs << " pairs rejected (< "
            << kMinMatchesPerPair << " matches), "
            << filter_stats.remaining_pairs << " pairs remaining with "
            << filter_stats.remaining_matches << " matches." << std::endl;

  GroundTruthContext gt;
  if (!LoadGroundTruth(gt_param_file, j, gt, err)) {
    std::cerr << err << std::endl;
    return 1;
  }
  if (gt.has_gt) {
    std::cout << "Ground truth loaded from: " << gt.source << std::endl;
  }

  if (frame_poses_file.empty()) {
    if (std::ifstream(kDefaultPosesPathA.c_str()).good()) {
      frame_poses_file = kDefaultPosesPathA;
    } else if (std::ifstream(kDefaultPosesPathB.c_str()).good()) {
      frame_poses_file = kDefaultPosesPathB;
    }
  }

  json input_poses_json;
  bool poses_loaded = false;
  if (!frame_poses_file.empty()) {
    if (LoadJsonFile(frame_poses_file, input_poses_json)) {
      poses_loaded = true;
      std::cout << "Loaded camera poses from: " << frame_poses_file << std::endl;
    } else {
      std::cerr << "Warning: Failed to parse frame poses file: " << frame_poses_file << std::endl;
    }
  }

  OptimizationCoordinator coordinator;
  if (gt.has_gt) {
    coordinator.SetGroundTruth(gt.camera);
  }
  if (poses_loaded) {
    coordinator.LoadFramePoses(input_poses_json);
  }

  OptimizationResult result = coordinator.RunIncrementalBA(input, config);
  if (!result.success) {
    std::cerr << "KITTI offline stereo BA did not pass the quality gate, writing best estimate anyway." << std::endl;
  }

  json out = BuildResultJson(result, true);
  out["experiment"] = {
      {"reset_camera_params_each_ba_round", config.reset_camera_params_each_ba_round},
  };
  AppendGroundTruthDiff(out, result, gt);

  std::ofstream fout(output_path.c_str());
  fout << out.dump(4) << std::endl;

  std::cout << "Result written to " << output_path << std::endl;
  return result.success ? 0 : 1;
}
