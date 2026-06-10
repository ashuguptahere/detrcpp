// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Differentiable bounding-box operations used by the matcher and the loss.
// Boxes are in two conventions: cxcywh (normalized center form, the model's
// output) and xyxy (corner form, for IoU). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/types.h>

#include <utility>

namespace detr::train {

// [..., 4] cxcywh -> [..., 4] xyxy.
torch::Tensor BoxCxcywhToXyxy(const torch::Tensor& x);

// [..., 4] xyxy -> [..., 4] cxcywh.
torch::Tensor BoxXyxyToCxcywh(const torch::Tensor& x);

// Area of xyxy boxes: [..., 4] -> [...].
torch::Tensor BoxArea(const torch::Tensor& boxes);

// Pairwise IoU between a:[N,4] and b:[M,4] (xyxy). Returns {iou:[N,M], union:[N,M]}.
std::pair<torch::Tensor, torch::Tensor> BoxIou(const torch::Tensor& a, const torch::Tensor& b);

// Pairwise generalized IoU between a:[N,4] and b:[M,4] (xyxy) -> [N,M].
torch::Tensor GeneralizedBoxIou(const torch::Tensor& a, const torch::Tensor& b);

// Generalized IoU of corresponding pairs a[i] vs b[i] (xyxy); a, b both [..., 4]
// -> [...]. Numerically equal to GeneralizedBoxIou(a, b).diagonal() but O(M)
// compute/memory instead of O(M^2) — for the loss, which only needs the diagonal.
torch::Tensor GeneralizedBoxIouPaired(const torch::Tensor& a, const torch::Tensor& b);

}  // namespace detr::train
