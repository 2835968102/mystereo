#include "stereo_io.h"


#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace stereocalib {

// ─── JSON serialization ─────────────────────────────────────────────────────

Intrinsics IntrinsicsFromJson(const nlohmann::json& j)
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

nlohmann::json IntrinsicsToJson(const Intrinsics& intr)
{
  return {{"fx", intr.fx}, {"fy", intr.fy}, {"cx", intr.cx}, {"cy", intr.cy},
          {"k1", intr.k1}, {"k2", intr.k2}, {"p1", intr.p1}, {"p2", intr.p2}, {"k3", intr.k3}};
}

StereoExtrinsics ExtrinsicsFromJson(const nlohmann::json& j)
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

nlohmann::json ExtrinsicsToJson(const StereoExtrinsics& ext)
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

// ─── Match data parsing ─────────────────────────────────────────────────────

std::vector<StereoPair> StereoPairsFromJson(const nlohmann::json& j)
{
  std::vector<StereoPair> pairs;
  for (const auto& jp : j) {
    StereoPair pair;
    pair.name = jp.value("name", "");
    for (const auto& jm : jp.at("matches")) {
      StereoMatch m;
      auto jl = jm.at("left");
      auto jr = jm.at("right");
      m.pt_left  = {jl[0].get<float>(), jl[1].get<float>()};
      m.pt_right = {jr[0].get<float>(), jr[1].get<float>()};
      pair.matches.push_back(m);
    }
    pairs.push_back(pair);
  }
  return pairs;
}

std::vector<RawImagePair> RawPairsFromJson(const nlohmann::json& j)
{
  std::vector<RawImagePair> pairs;
  for (size_t i = 0; i < j.size(); ++i) {
    const nlohmann::json& jp = j.at(i);

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
      const nlohmann::json& jm_list = jp.at("matches");
      for (size_t k = 0; k < jm_list.size(); ++k) {
        const nlohmann::json& jm = jm_list.at(k);
        if (!jm.contains("left") || !jm.contains("right")) {
          continue;
        }

        const nlohmann::json& ja = jm.at("left");
        const nlohmann::json& jb = jm.at("right");
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

// ─── Camera parameter file loading ──────────────────────────────────────────

namespace {

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

  nlohmann::json j;
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

}  // namespace

bool LoadCameraFromFile(const std::string& path, StereoCamera& cam, std::string& err)
{
  const size_t dot = path.find_last_of('.');
  const std::string ext = (dot == std::string::npos) ? "" : path.substr(dot + 1);
  if (ext == "json") {
    return LoadCameraFromJsonFile(path, cam, err);
  }
  return LoadInitCameraFromText(path, cam, err);
}

}  // namespace stereocalib
