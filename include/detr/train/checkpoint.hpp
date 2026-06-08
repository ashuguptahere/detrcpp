// Copyright 2026 detrcpp authors. Apache-2.0.
//
// CheckpointMgr writes resumable checkpoints. For a tag (e.g. "last" / "best")
// it saves three files in the run directory:
//   <tag>.safetensors        EMA weights (primary — what we ship/evaluate)
//   <tag>.model.safetensors  raw live weights (to resume the exact model)
//   <tag>.opt                optimizer state (AdamW moments) for resume
// Training metadata (epoch, step, seed, best metric) rides in the safetensors
// __metadata__ block, so a checkpoint is self-describing and the EMA file is a
// plain .safetensors any upstream repo can load. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <filesystem>
#include <string>

#include <torch/torch.h>

#include "detr/core/result.hpp"
#include "detr/models/model.hpp"
#include "detr/train/ema.hpp"

namespace detr::train {

struct TrainState {
  int epoch{0};
  long long global_step{0};
  unsigned long long seed{0};
  double best_metric{-1.0};
};

class CheckpointMgr {
 public:
  explicit CheckpointMgr(std::filesystem::path dir);

  core::Result<void> Save(const std::string& tag, models::IModel& model, const ModelEma& ema,
                          torch::optim::Optimizer& opt, const TrainState& state);

  core::Result<TrainState> Load(const std::string& tag, models::IModel& model, ModelEma& ema,
                                torch::optim::Optimizer& opt);

  const std::filesystem::path& dir() const { return dir_; }

 private:
  std::filesystem::path dir_;
};

}  // namespace detr::train
