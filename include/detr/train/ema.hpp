// Copyright 2026 detrcpp authors. Apache-2.0.
//
// ModelEma: an exponential moving average of a model's parameters. The averaged
// ("shadow") weights are what we evaluate and ship as last/best checkpoints —
// they are consistently a bit better and far less noisy than the raw weights.
// shadow <- decay * shadow + (1 - decay) * param, each optimizer step.

#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

#include "detr/weights/state_dict.hpp"

namespace detr::train {

class ModelEma {
 public:
  ModelEma(const torch::nn::Module& model, double decay = 0.9999);

  // Folds the model's current parameters into the moving average.
  void Update(const torch::nn::Module& model);

  // Writes the averaged weights into |model| (e.g. before eval / export).
  void CopyTo(torch::nn::Module& model) const;

  // The shadow weights as a StateDict (named like the model) for checkpointing.
  weights::StateDict State() const;

  // Restores shadow weights from a StateDict (resume).
  void LoadState(const weights::StateDict& state);

  double decay() const { return decay_; }

 private:
  double decay_;
  std::vector<std::string> names_;
  std::vector<torch::Tensor> shadow_;  // detached CPU/Device copies, by index
};

}  // namespace detr::train
