#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <map>
#include <limits>

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

std::string Trim(const std::string& s)
{
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
    ++b;
  }
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
    --e;
  }
  return s.substr(b, e - b);
}

bool LoadInitCameraFromText(const std::string& path, StereoCamera& cam, std::string& err)
{
  std::ifstream fin(path.c_str());
  if (!fin.is_open()) {
    err = "Cannot open init param file: " + path;
    return false;
  }

  std::map<std::string, std::string> kv;
  std::string line;
  int line_no = 0;
  while (std::getline(fin, line)) {
    ++line_no;
    const std::string s = Trim(line);
    if (s.empty() || s[0] == '#') {
      continue;
    }
    const size_t eq = s.find('=');
    if (eq == std::string::npos) {
      err = "Invalid line " + std::to_string(line_no) + " in init param file (expect key=value).";
      return false;
    }
    const std::string key = Trim(s.substr(0, eq));
    const std::string val = Trim(s.substr(eq + 1));
    if (key.empty() || val.empty()) {
      err = "Invalid line " + std::to_string(line_no) + " in init param file (empty key/value).";
      return false;
    }
    kv[key] = val;
  }

  auto get_double = [&](const std::string& key, double& out) -> bool {
    const std::map<std::string, std::string>::const_iterator it = kv.find(key);
    if (it == kv.end()) {
      err = "Missing key in init param file: " + key;
      return false;
    }
    try {
      out = std::stod(it->second);
    } catch (...) {
      err = "Invalid numeric value for key: " + key;
      return false;
    }
    return true;
  };

  if (!get_double("left_fx", cam.left.fx) || !get_double("left_fy", cam.left.fy) ||
      !get_double("left_cx", cam.left.cx) || !get_double("left_cy", cam.left.cy) ||
      !get_double("left_k1", cam.left.k1) || !get_double("left_k2", cam.left.k2) ||
      !get_double("left_p1", cam.left.p1) || !get_double("left_p2", cam.left.p2) ||
      !get_double("left_k3", cam.left.k3)) {
    return false;
  }

  if (!get_double("right_fx", cam.right.fx) || !get_double("right_fy", cam.right.fy) ||
      !get_double("right_cx", cam.right.cx) || !get_double("right_cy", cam.right.cy) ||
      !get_double("right_k1", cam.right.k1) || !get_double("right_k2", cam.right.k2) ||
      !get_double("right_p1", cam.right.p1) || !get_double("right_p2", cam.right.p2) ||
      !get_double("right_k3", cam.right.k3)) {
    return false;
  }

  double r00, r01, r02, r10, r11, r12, r20, r21, r22;
  if (!get_double("R00", r00) || !get_double("R01", r01) || !get_double("R02", r02) ||
      !get_double("R10", r10) || !get_double("R11", r11) || !get_double("R12", r12) ||
      !get_double("R20", r20) || !get_double("R21", r21) || !get_double("R22", r22)) {
    return false;
  }
  cam.extrinsics.R = (cv::Mat_<double>(3, 3) << r00, r01, r02, r10, r11, r12, r20, r21, r22);

  double tx, ty, tz;
  if (!get_double("tx", tx) || !get_double("ty", ty) || !get_double("tz", tz)) {
    return false;
  }
  cam.extrinsics.t = (cv::Mat_<double>(3, 1) << tx, ty, tz);

  return true;
}

bool LoadCameraFromJsonFile(const std::string& path, StereoCamera& cam, std::string& err)
{
  std::ifstream fin(path.c_str());
  if (!fin.is_open()) {
    err = "Cannot open camera json file: " + path;
    return false;
  }

  json j;
  try {
    fin >> j;
  } catch (...) {
    err = "Invalid json file: " + path;
    return false;
  }

  if (!j.contains("left") || !j.contains("right") || !j.contains("extrinsics")) {
    err = "Camera json must contain left/right/extrinsics: " + path;
    return false;
  }

  try {
    cam.left = IntrinsicsFromJson(j.at("left"));
    cam.right = IntrinsicsFromJson(j.at("right"));
    cam.extrinsics = ExtrinsicsFromJson(j.at("extrinsics"));
  } catch (...) {
    err = "Failed to parse camera json fields: " + path;
    return false;
  }
  return true;
}

bool LoadCameraFromFile(const std::string& path, StereoCamera& cam, std::string& err)
{
  const size_t dot = path.find_last_of('.');
  const std::string ext = (dot == std::string::npos) ? "" : path.substr(dot + 1);
  if (ext == "json") {
    return LoadCameraFromJsonFile(path, cam, err);
  }
  return LoadInitCameraFromText(path, cam, err);
}

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

json IntrinsicsDiffToJson(const Intrinsics& est, const Intrinsics& gt)
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

json ExtrinsicsDiffToJson(const StereoExtrinsics& est, const StereoExtrinsics& gt)
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

}  // namespace

int main(int argc, char** argv)
{
  std::string input_path;
  std::string output_path;
  std::string gt_param_file;
  const std::string kForcedInitPathA = "stereo_calib/data/example_init_params.txt";
  const std::string kForcedInitPathB = "../stereo_calib/data/example_init_params.txt";

  OfflineStereoBA::Options options;

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
    }
  }

  if (input_path.empty() || output_path.empty()) {
    std::cerr << "Usage: run_offline_stereo_ba --input <matches.json> --output <result.json> "
              << "[--gt_param_file gt_params.{txt|json}] "
              << "[--max_iter 200] [--incremental_max_iter 20] [--global_opt_interval 5] "
              << "[--min_track_len 3] [--huber 1.0] [--max_score 1.0] "
              << "[--min_pair_inliers 12] [--min_pair_inlier_ratio 0.35] "
              << "[--fix_distortion] [--aspect_ratio_prior 1.0] "
              << "[--baseline_prior 10.0] [--max_reproj_error 20.0]\n"
              << "Initial values are always loaded from stereo_calib/example_init_params.txt"
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
    if (!j.contains("pairs")) {
      std::cerr << "Input json must contain: pairs" << std::endl;
      return 1;
    }
  }

  OfflineBAInput input;
  input.pairs = PairsFromJson(j.at("pairs"));

  std::string err;
  if (!LoadInitCameraFromText(kForcedInitPathA, input.init_camera, err)) {
    if (!LoadInitCameraFromText(kForcedInitPathB, input.init_camera, err)) {
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

  StereoCamera gt_camera;
  bool has_gt = false;
  std::string gt_source;
  if (!gt_param_file.empty()) {
    std::string err;
    if (!LoadCameraFromFile(gt_param_file, gt_camera, err)) {
      std::cerr << err << std::endl;
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
