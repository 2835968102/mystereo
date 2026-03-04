#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "offline_stereo_ba.h"

using json = nlohmann::json;
using namespace stereocalib;

namespace {

Intrinsics IntrinsicsFromJson(const json& j)
{
  Intrinsics intr;
  intr.fx = j.at("fx").get<double>();
  intr.fy = j.at("fy").get<double>();
  intr.cx = j.at("cx").get<double>();
  intr.cy = j.at("cy").get<double>();
  intr.k1 = j.value("k1", 0.0);
  intr.k2 = j.value("k2", 0.0);
  intr.p1 = j.value("p1", 0.0);
  intr.p2 = j.value("p2", 0.0);
  intr.k3 = j.value("k3", 0.0);
  return intr;
}

json IntrinsicsToJson(const Intrinsics& intr)
{
  return {
      {"fx", intr.fx}, {"fy", intr.fy}, {"cx", intr.cx}, {"cy", intr.cy},
      {"k1", intr.k1}, {"k2", intr.k2}, {"p1", intr.p1}, {"p2", intr.p2}, {"k3", intr.k3},
  };
}

StereoExtrinsics ExtrinsicsFromJson(const json& j)
{
  StereoExtrinsics ext;
  const std::vector<double> R_vec = j.at("R").get<std::vector<double> >();
  const std::vector<double> t_vec = j.at("t").get<std::vector<double> >();

  ext.R = (cv::Mat_<double>(3, 3) << R_vec[0], R_vec[1], R_vec[2],
                                      R_vec[3], R_vec[4], R_vec[5],
                                      R_vec[6], R_vec[7], R_vec[8]);
  ext.t = (cv::Mat_<double>(3, 1) << t_vec[0], t_vec[1], t_vec[2]);
  return ext;
}

json ExtrinsicsToJson(const StereoExtrinsics& ext)
{
  std::vector<double> R_vec(9, 0.0);
  std::vector<double> t_vec(3, 0.0);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R_vec[r * 3 + c] = ext.R.at<double>(r, c);
    }
  }
  for (int i = 0; i < 3; ++i) {
    t_vec[i] = ext.t.at<double>(i, 0);
  }
  return {{"R", R_vec}, {"t", t_vec}};
}

std::vector<RawImagePair> PairsFromJson(const json& j)
{
  std::vector<RawImagePair> pairs;
  for (size_t i = 0; i < j.size(); ++i) {
    const json& jp = j.at(i);

    std::string image_a;
    std::string image_b;

    if (jp.contains("left_image") && jp.contains("right_image")) {
      image_a = jp.at("left_image").get<std::string>();
      image_b = jp.at("right_image").get<std::string>();
    } else if (jp.contains("image_a") && jp.contains("image_b")) {
      image_a = jp.at("image_a").get<std::string>();
      image_b = jp.at("image_b").get<std::string>();
    } else {
      continue;
    }

    RawImagePair pair;
    pair.image_a = image_a;
    pair.image_b = image_b;

    if (jp.contains("matches") && jp.at("matches").is_array()) {
      const json& jm_list = jp.at("matches");
      for (size_t k = 0; k < jm_list.size(); ++k) {
        const json& jm = jm_list.at(k);
        if (!jm.contains("left") || !jm.contains("right")) {
          continue;
        }

        const json& ja = jm.at("left");
        const json& jb = jm.at("right");
        if (ja.size() < 2 || jb.size() < 2) {
          continue;
        }

        RawPairMatch m;
        m.pt_a = cv::Point2f(ja.at(0).get<float>(), ja.at(1).get<float>());
        m.pt_b = cv::Point2f(jb.at(0).get<float>(), jb.at(1).get<float>());
        m.score = jm.value("score", 0.0);
        pair.matches.push_back(m);
      }
    }

    pairs.push_back(pair);
  }
  return pairs;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string input_path;
  std::string output_path;

  OfflineStereoBA::Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--max_iter" && i + 1 < argc) {
      options.max_iter = std::stoi(argv[++i]);
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
    } else if (arg == "--max_reproj_error" && i + 1 < argc) {
      options.max_reproj_error = std::stod(argv[++i]);
    } else if (arg == "--baseline_prior" && i + 1 < argc) {
      options.baseline_prior_weight = std::stod(argv[++i]);
    }
  }

  if (input_path.empty() || output_path.empty()) {
    std::cerr << "Usage: run_offline_stereo_ba --input <matches.json> --output <result.json> "
              << "[--max_iter 200] [--min_track_len 3] [--huber 1.0] [--max_score 1.0] "
              << "[--min_pair_inliers 12] [--min_pair_inlier_ratio 0.35] "
              << "[--fix_distortion] [--aspect_ratio_prior 1.0] "
              << "[--baseline_prior 10.0] [--max_reproj_error 20.0]"
              << std::endl;
    return 1;
  }

  std::ifstream fin(input_path.c_str());
  if (!fin.is_open()) {
    std::cerr << "Cannot open input file: " << input_path << std::endl;
    return 1;
  }

  json j;
  fin >> j;

  if (!j.contains("left") || !j.contains("right") || !j.contains("extrinsics") || !j.contains("pairs")) {
    std::cerr << "Input json must contain: left, right, extrinsics, pairs" << std::endl;
    return 1;
  }

  OfflineBAInput input;
  input.init_camera.left = IntrinsicsFromJson(j.at("left"));
  input.init_camera.right = IntrinsicsFromJson(j.at("right"));
  input.init_camera.extrinsics = ExtrinsicsFromJson(j.at("extrinsics"));
  input.pairs = PairsFromJson(j.at("pairs"));

  std::size_t raw_matches = 0;
  for (std::size_t i = 0; i < input.pairs.size(); ++i) {
    raw_matches += input.pairs[i].matches.size();
  }

  std::cout << "Loaded " << input.pairs.size() << " pair records, "
            << raw_matches << " raw matches." << std::endl;

  OfflineStereoBA optimizer(input, options);
  StereoCamera result_camera;
  bool success = optimizer.Solve(result_camera);

  if (!success) {
    std::cerr << "Offline stereo BA did not pass the quality gate, writing best estimate anyway." << std::endl;
  }

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

  std::ofstream fout(output_path.c_str());
  fout << out.dump(4) << std::endl;

  std::cout << "Result written to " << output_path << std::endl;
  return success ? 0 : 1;
}
