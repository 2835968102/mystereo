#ifndef STEREO_CALIB_SRC_SERVICES_EVALUATION_SERVICE_H
#define STEREO_CALIB_SRC_SERVICES_EVALUATION_SERVICE_H

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "stereo_types.h"

namespace stereocalib {

// ─── Evaluation Service Interface ───────────────────────────────────────────

class IEvaluationService {
 public:
  virtual ~IEvaluationService() = default;

  /// Set ground truth for comparison.
  virtual void SetGroundTruth(const StereoCamera& gt) = 0;
  
  /// Check if ground truth is available.
  virtual bool HasGroundTruth() const = 0;

  /// Print current parameters vs ground truth.
  /// @param stage_name Name of the optimization stage.
  /// @param current Current camera parameters.
  virtual void PrintCurrentVsGroundTruth(const std::string& stage_name,
                                         const StereoCamera& current) const = 0;

  /// Record an optimization stage in history.
  /// @param stage_name Name of the optimization stage.
  /// @param reproj_error Current reprojection error.
  /// @param current Current camera parameters.
  virtual void RecordOptimizationStage(const std::string& stage_name,
                                       double reproj_error,
                                       const StereoCamera& current) = 0;

  /// Get the recorded optimization history.
  virtual std::vector<nlohmann::json> GetOptimizationHistory() const = 0;
  
  /// Clear optimization history.
  virtual void ClearHistory() = 0;
};

// ─── Evaluation Service Implementation ──────────────────────────────────────

class EvaluationService : public IEvaluationService {
 public:
  EvaluationService() = default;
  ~EvaluationService() override = default;

  void SetGroundTruth(const StereoCamera& gt) override;
  bool HasGroundTruth() const override;

  void PrintCurrentVsGroundTruth(const std::string& stage_name,
                                 const StereoCamera& current) const override;

  void RecordOptimizationStage(const std::string& stage_name,
                               double reproj_error,
                               const StereoCamera& current) override;

  std::vector<nlohmann::json> GetOptimizationHistory() const override;
  void ClearHistory() override;

 private:
  bool has_ground_truth_ = false;
  StereoCamera ground_truth_;
  std::vector<nlohmann::json> optimization_history_;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_SERVICES_EVALUATION_SERVICE_H
