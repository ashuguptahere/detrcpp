// Copyright 2026 detrcpp authors. Apache-2.0.
//
// HungarianMatcher: DETR's bipartite matching between the Q predictions and the
// T ground-truth objects of each image. The match cost combines classification
// probability, L1 box distance, and (negative) GIoU; the optimal assignment is
// found with the core LinearSumAssignment solver. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/types.h>

#include <utility>
#include <vector>

#include "detr/models/model.hpp"
#include "detr/train/target.hpp"

namespace detr::train {

struct MatchWeights {
  double cls{1.0};
  double bbox{5.0};
  double giou{2.0};
  bool focal{false};  // sigmoid-focal classification cost (vs softmax prob).
  double focal_alpha{0.25};
  double focal_gamma{2.0};
};

// One matched index pair per image: src (query indices) and tgt (target
// indices), both [K] int64 where K = number of objects in that image.
using MatchIndices = std::pair<torch::Tensor, torch::Tensor>;

// Computes the optimal assignment for every image in the batch. No gradients
// flow through matching (it only selects indices).
std::vector<MatchIndices> HungarianMatch(const models::Detections& outputs,
                                         const TargetBatch& targets,
                                         const MatchWeights& weights = {});

// One-to-many assignment for RT-DETRv3 hierarchical dense positive supervision:
// each ground-truth object keeps its |k| lowest-cost queries (so every GT appears
// k times in the returned tgt indices). No gradients flow through matching.
std::vector<MatchIndices> OneToManyMatch(const models::Detections& outputs,
                                         const TargetBatch& targets, int k,
                                         const MatchWeights& weights = {});

}  // namespace detr::train
