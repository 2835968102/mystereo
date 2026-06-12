#pragma once

#include <ceres/ceres.h>

#include <vector>

namespace stereocalib {

inline void SetSubsetParameterBlock(
    ceres::Problem& problem,
    double* parameters,
    int parameter_count,
    const std::vector<int>& fixed_indices) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
  problem.SetManifold(
      parameters,
      new ceres::SubsetManifold(parameter_count, fixed_indices));
#else
  problem.SetParameterization(
      parameters,
      new ceres::SubsetParameterization(parameter_count, fixed_indices));
#endif
}

}  // namespace stereocalib
