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
  double eos_coef{0.1};       // softmax: down-weight of the "no object" class.
  double focal_alpha{0.25};   // sigmoid-focal models only.
  double focal_gamma{2.0};
};

struct Losses {
  torch::Tensor loss_ce;
  torch::Tensor loss_bbox;
  torch::Tensor loss_giou;
  torch::Tensor total;  // weighted sum (the value to backprop).
};

class SetCriterion {
 public:
  SetCriterion(int num_classes, LossWeights weights, bool focal = false)
      : num_classes_(num_classes), w_(weights), focal_(focal) {}

  Losses Compute(const models::Detections& outputs, const TargetBatch& targets,
                 const std::vector<MatchIndices>& matches) const;

 private:
  int num_classes_;  // background class id = num_classes_ (last softmax logit).
  LossWeights w_;
  bool focal_;       // sigmoid focal classification (no no-object slot).
};

}  // namespace detr::train
