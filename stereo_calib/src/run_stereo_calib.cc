/*
 * run_stereo_calib – entry point for stereo camera parameter optimisation.
 *
 * Usage:
 *   run_stereo_calib --input <matches.json> --output <result.json> [--max_iter 200]
 *
 * Input JSON format:
 * {
 *   "left":  { "fx":..., "fy":..., "cx":..., "cy":...,
 *              "k1":..., "k2":..., "p1":..., "p2":..., "k3":... },
 *   "right": { <same fields> },
 *   "extrinsics": {
 *     "R": [r00,r01,r02, r10,r11,r12, r20,r21,r22],   // row-major 3×3
 *     "t": [tx, ty, tz]
 *   },
 *   "pairs": [
 *     {
 *       "name": "pair_001",
 *       "matches": [
 *         { "left": [u_l, v_l], "right": [u_r, v_r] },
 *         ...
 *       ]
 *     },
 *     ...
 *   ]
 * }
 *
 * Output JSON format:  same structure as input, with refined values.
 */

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "stereo_optimizer.h"

using json = nlohmann::json;
using namespace stereocalib;
using namespace std;

// ─── JSON helpers ─────────────────────────────────────────────────────────────

static Intrinsics IntrinsicsFromJson(const json& j)
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

static json IntrinsicsToJson(const Intrinsics& intr)
{
  return {{"fx", intr.fx}, {"fy", intr.fy}, {"cx", intr.cx}, {"cy", intr.cy},
          {"k1", intr.k1}, {"k2", intr.k2}, {"p1", intr.p1}, {"p2", intr.p2}, {"k3", intr.k3}};
}

static StereoExtrinsics ExtrinsicsFromJson(const json& j)
{
  StereoExtrinsics ext;
  vector<double> R_vec = j.at("R").get<vector<double>>();
  vector<double> t_vec = j.at("t").get<vector<double>>();

  ext.R = (cv::Mat_<double>(3, 3) << R_vec[0], R_vec[1], R_vec[2],
                                      R_vec[3], R_vec[4], R_vec[5],
                                      R_vec[6], R_vec[7], R_vec[8]);
  ext.t = (cv::Mat_<double>(3, 1) << t_vec[0], t_vec[1], t_vec[2]);
  return ext;
}

static json ExtrinsicsToJson(const StereoExtrinsics& ext)
{
  vector<double> R_vec(9), t_vec(3);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R_vec[r * 3 + c] = ext.R.at<double>(r, c);
  t_vec[0] = ext.t.at<double>(0);
  t_vec[1] = ext.t.at<double>(1);
  t_vec[2] = ext.t.at<double>(2);
  return {{"R", R_vec}, {"t", t_vec}};
}

static vector<StereoPair> PairsFromJson(const json& j)
{
  vector<StereoPair> pairs;
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

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  // ── Argument parsing (simple) ──────────────────────────────────────────────
  string input_path, output_path;
  int    max_iter         = 200;
  double max_reproj_error = 3.0;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--input"  && i + 1 < argc) input_path       = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--max_iter" && i + 1 < argc) max_iter  = stoi(argv[++i]);
    else if (arg == "--max_reproj_error" && i + 1 < argc) max_reproj_error = stod(argv[++i]);
  }

  if (input_path.empty() || output_path.empty()) {
    cerr << "Usage: run_stereo_calib --input <matches.json> --output <result.json>"
         << " [--max_iter 200] [--max_reproj_error 3.0]" << endl;
    return 1;
  }

  // ── Read input ─────────────────────────────────────────────────────────────
  ifstream fin(input_path);
  if (!fin.is_open()) {
    cerr << "Cannot open input file: " << input_path << endl;
    return 1;
  }
  json j_in;
  fin >> j_in;

  StereoCamera init_camera;
  init_camera.left       = IntrinsicsFromJson(j_in.at("left"));
  init_camera.right      = IntrinsicsFromJson(j_in.at("right"));
  init_camera.extrinsics = ExtrinsicsFromJson(j_in.at("extrinsics"));

  vector<StereoPair> pairs = PairsFromJson(j_in.at("pairs"));

  size_t total_matches = 0;
  for (const auto& p : pairs) total_matches += p.matches.size();
  cout << "Loaded " << pairs.size() << " pair(s), " << total_matches << " matches total." << endl;

  // ── Optimise ───────────────────────────────────────────────────────────────
  StereoOptimizer optimizer(pairs, init_camera, max_iter, max_reproj_error);
  StereoCamera    result_camera;
  bool success = optimizer.Solve(result_camera);

  if (!success) {
    cerr << "Optimisation failed." << endl;
    // Still write current best result for inspection
    result_camera = init_camera;
  }

  // ── Write output ───────────────────────────────────────────────────────────
  json j_out;
  j_out["left"]         = IntrinsicsToJson(result_camera.left);
  j_out["right"]        = IntrinsicsToJson(result_camera.right);
  j_out["extrinsics"]   = ExtrinsicsToJson(result_camera.extrinsics);
  j_out["init_reproj_error"]  = optimizer.init_reproj_error();
  j_out["final_reproj_error"] = optimizer.final_reproj_error();
  j_out["success"]            = success;

  ofstream fout(output_path);
  fout << j_out.dump(4) << endl;
  cout << "Result written to " << output_path << endl;

  return success ? 0 : 1;
}
