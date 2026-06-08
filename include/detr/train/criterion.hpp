// Copyright 2026 detrcpp authors. Apache-2.0.
//
// SetCriterion: DETR's set-prediction loss. Given the matcher's assignment, it
// computes (1) classification cross-entropy over all queries — matched queries
// get their target class, the rest get the "no object" class with a reduced
// weight (eos_coef) — plus (2) an L1 box loss and (3) a GIoU loss over the
// matched pairs, each normalized by the number of target boxes. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <vector>

#include <torch/torch.h>

#include "detr/models/model.hpp"
#include "detr/train/matcher.hpp"
#include "detr/train/target.hpp"

namespace detr::train {

struct LossWeights {
  double cls{1.0};
  double bbox{5.0};
  double giou{2.0};
  double eos_coef{0.1};  // down-weight of the "no object" class.
};

struct Losses {
  torch::Tensor loss_ce;
  torch::Tensor loss_bbox;
  torch::Tensor loss_giou;
  torch::Tensor total;  // weighted sum (the value to backprop).
};

class SetCriterion {
 public:
  SetCriterion(int num_classes, LossWeights weights) : num_classes_(num_classes), w_(weights) {}

  Losses Compute(const models::Detections& outputs, const TargetBatch& targets,
                 const std::vector<MatchIndices>& matches) const;

 private:
  int num_classes_;  // the "no object" class id is num_classes_ (the last logit).
  LossWeights w_;
};

}  // namespace detr::train
