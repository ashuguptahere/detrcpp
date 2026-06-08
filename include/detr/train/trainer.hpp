// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Trainer: ties a model, an AdamW optimizer, the Hungarian matcher, the set
// criterion, and an EMA together into a single optimization step. The loop over
// epochs/batches and checkpointing live in the caller (CLI/Fit); this keeps the
// step testable in isolation. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <cstdint>
#include <memory>

#include <torch/torch.h>

#include "detr/models/model.hpp"
#include "detr/train/criterion.hpp"
#include "detr/train/ema.hpp"
#include "detr/train/matcher.hpp"
#include "detr/train/target.hpp"

namespace detr::train {

struct TrainConfig {
  int epochs{300};
  double lr{1e-4};
  double weight_decay{1e-4};
  double grad_clip{0.1};  // max grad norm; 0 disables.
  std::uint64_t seed{0};
  double ema_decay{0.9999};
  MatchWeights match{};
  LossWeights loss{};
};

class Trainer {
 public:
  Trainer(std::shared_ptr<models::IModel> model, TrainConfig cfg);

  // One optimization step on a batch: forward -> match -> loss -> backward ->
  // (clip) -> step -> EMA update. Returns the total loss value.
  float TrainStep(const torch::Tensor& images, const TargetBatch& targets);

  models::IModel& Model() { return *model_; }
  ModelEma& Ema() { return ema_; }
  torch::optim::Optimizer& Optimizer() { return *opt_; }
  const TrainConfig& Config() const { return cfg_; }

 private:
  std::shared_ptr<models::IModel> model_;
  TrainConfig cfg_;
  SetCriterion criterion_;
  ModelEma ema_;
  std::unique_ptr<torch::optim::AdamW> opt_;
};

}  // namespace detr::train
