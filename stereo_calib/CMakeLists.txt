cmake_minimum_required(VERSION 3.10)
project(StereoCalib)

# ── Build settings ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-g -Wall -pthread -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────

# nlohmann/json  (reuse from parent project's 3rdparty directory)
set(PARENT_3RDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty)
include_directories(${PARENT_3RDPARTY_DIR}/json-3.9.1/single_include)

# OpenCV
find_package(OpenCV 4.3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
include_directories("${CMAKE_INSTALL_PREFIX}/include/eigen3")

set(EXTERNAL_LIBRARIES
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)

# ── Sources ───────────────────────────────────────────────────────────────────
include_directories(src)

set(STEREO_CORE_SRCS
    src/services/track_service.cc
    src/services/initialization_service.cc
    src/services/bundle_adjustment_service.cc
    src/services/outlier_rejection_service.cc
    src/services/evaluation_service.cc
    src/core/camera_math.cc
    src/stereo_factors.cc
    src/stereo_io.cc
    src/stereo_eval.cc
    src/track_builder.cc
    src/stereo_optimizer.cc
    src/offline_stereo_ba.cc
)

# Core library (reusable by tests, other apps)
add_library(stereo_core ${STEREO_CORE_SRCS})
target_link_libraries(stereo_core ${EXTERNAL_LIBRARIES})

# Main executable
add_executable(run_stereo_calib src/run_stereo_calib.cc)
target_link_libraries(run_stereo_calib stereo_core ${EXTERNAL_LIBRARIES})

# Offline BA executable (track-building + global robust BA)
add_executable(run_offline_stereo_ba src/run_offline_stereo_ba.cc)
target_link_libraries(run_offline_stereo_ba stereo_core ${EXTERNAL_LIBRARIES})
