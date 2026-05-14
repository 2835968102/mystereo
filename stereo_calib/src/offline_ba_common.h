#ifndef STEREO_CALIB_SRC_OFFLINE_BA_COMMON_H
#define STEREO_CALIB_SRC_OFFLINE_BA_COMMON_H

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "coordinators/optimization_coordinator.h"
#include "stereo_types.h"

namespace stereocalib {

struct PairFilterStats {
  std::size_t rejected_pairs = 0;
  std::size_t remaining_pairs = 0;
  std::size_t remaining_matches = 0;
};

struct GroundTruthContext {
  StereoCamera camera;
  bool has_gt = false;
  std::string source;
};

bool LoadJsonFile(const std::string& path, nlohmann::json& parsed);

bool LoadOfflineBAInputJson(const std::string& input_path,
                            nlohmann::json& parsed,
                            OfflineBAInput& input,
                            std::string& err);

std::size_t CountRawMatches(const std::vector<RawImagePair>& pairs);

PairFilterStats FilterPairsByMinMatches(std::vector<RawImagePair>& pairs,
                                         int min_matches_per_pair);

bool LoadGroundTruth(const std::string& gt_param_file,
                     const nlohmann::json& input_json,
                     GroundTruthContext& gt,
                     std::string& err);

nlohmann::json BuildSummaryFromHistory(const nlohmann::json& history);

nlohmann::json BuildResultJson(const OptimizationResult& result,
                               bool include_conflict_stats);

void AppendGroundTruthDiff(nlohmann::json& out,
                           const OptimizationResult& result,
                           const GroundTruthContext& gt);

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_OFFLINE_BA_COMMON_H

