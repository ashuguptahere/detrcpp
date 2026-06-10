// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Per-image ground truth for set prediction. labels: [T] int64 class ids;
// boxes: [T,4] float32 in normalized cxcywh. A batch is a vector<Target> (one
// per image), because images have different object counts.

#pragma once

#include <torch/types.h>

#include <vector>

namespace detr::train {

struct Target {
  torch::Tensor labels;  // [T] int64
  torch::Tensor boxes;   // [T, 4] float32, normalized cxcywh
};

using TargetBatch = std::vector<Target>;

}  // namespace detr::train
