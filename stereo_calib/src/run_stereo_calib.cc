/*
 * run_stereo_calib – entry point for stereo camera parameter optimisation.
 *
 * Usage:
 *   run_stereo_calib --input <matches.json> --output <result.json> [--max_iter 200]
 */

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "stereo_io.h"
#include "stereo_optimizer.h"

using json = nlohmann::json;
using namespace stereocalib;
using namespace std;

int main(int argc, char** argv)
{
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

  vector<StereoPair> pairs = StereoPairsFromJson(j_in.at("pairs"));

  size_t total_matches = 0;
  for (const auto& p : pairs) total_matches += p.matches.size();
  cout << "Loaded " << pairs.size() << " pair(s), " << total_matches << " matches total." << endl;

  // ── Optimise ───────────────────────────────────────────────────────────────
  StereoOptimizer optimizer(pairs, init_camera, max_iter, max_reproj_error);
  StereoCamera    result_camera;
  bool success = optimizer.Solve(result_camera);

  if (!success) {
    cerr << "Optimisation failed." << endl;
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
