/*
 * JSON serialization and file I/O for stereo camera parameters.
 */

#ifndef STEREO_CALIB_SRC_STEREO_IO_H
#define STEREO_CALIB_SRC_STEREO_IO_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "stereo_types.h"

namespace stereocalib {

// ─── StereoCamera JSON serialization ─────────────────────────────────────────

Intrinsics IntrinsicsFromJson(const nlohmann::json& j);
nlohmann::json IntrinsicsToJson(const Intrinsics& intr);

StereoExtrinsics ExtrinsicsFromJson(const nlohmann::json& j);
nlohmann::json ExtrinsicsToJson(const StereoExtrinsics& ext);

// ─── Match data parsing ──────────────────────────────────────────────────────

std::vector<StereoPair> StereoPairsFromJson(const nlohmann::json& j);
std::vector<RawImagePair> RawPairsFromJson(const nlohmann::json& j);

// ─── Camera parameter file loading ──────────────────────────────────────────

bool LoadCameraFromFile(const std::string& path, StereoCamera& cam, std::string& err);

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_IO_H
