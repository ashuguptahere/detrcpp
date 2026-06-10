// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/matcher.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "detr/core/assignment.hpp"
#include "detr/train/box_ops.hpp"

namespace detr::train {

namespace {

// The [Q, T] match cost (classification + L1 + GIoU), shared by the one-to-one
// Hungarian matcher and the one-to-many (dense-supervision) matcher.
torch::Tensor CostMatrix(const torch::Tensor& logits, const torch::Tensor& boxes,
                         const torch::Tensor& labels_in, const torch::Tensor& tgt_boxes_in,
                         const MatchWeights& weights) {
  auto labels = labels_in.to(logits.device());      // [T]
  auto tgt_boxes = tgt_boxes_in.to(boxes.device());  // [T, 4]
  torch::Tensor cost_class;
  if (weights.focal) {
    auto p = logits.sigmoid();  // [Q, num_classes]
    auto neg = (1 - weights.focal_alpha) * p.pow(weights.focal_gamma) * (-(1 - p + 1e-8).log());
    auto pos = weights.focal_alpha * (1 - p).pow(weights.focal_gamma) * (-(p + 1e-8).log());
    cost_class = pos.index_select(1, labels) - neg.index_select(1, labels);  // [Q, T]
  } else {
    cost_class = -logits.softmax(-1).index_select(1, labels);  // [Q, T]
  }
  auto cost_bbox = torch::cdist(boxes, tgt_boxes, /*p=*/1);  // [Q, T]
  auto cost_giou =
      -GeneralizedBoxIou(BoxCxcywhToXyxy(boxes), BoxCxcywhToXyxy(tgt_boxes));  // [Q, T]
  return weights.bbox * cost_bbox + weights.cls * cost_class + weights.giou * cost_giou;
}

}  // namespace

std::vector<MatchIndices> HungarianMatch(const models::Detections& outputs,
                                         const TargetBatch& targets, const MatchWeights& weights) {
  torch::NoGradGuard no_grad;
  const auto batch = outputs.logits.size(0);
  std::vector<MatchIndices> result;
  result.reserve(static_cast<std::size_t>(batch));

  const auto idx_opts = torch::TensorOptions().dtype(torch::kInt64);

  for (std::int64_t b = 0; b < batch; ++b) {
    const auto t = targets[static_cast<std::size_t>(b)];
    const auto num_targets = t.labels.size(0);
    if (num_targets == 0) {
      result.emplace_back(torch::empty({0}, idx_opts), torch::empty({0}, idx_opts));
      continue;
    }

    auto cost = CostMatrix(outputs.logits[b], outputs.boxes[b], t.labels, t.boxes, weights);
    cost = cost.to(torch::kCPU, torch::kDouble).contiguous();  // [Q, T]

    const auto q = static_cast<int>(cost.size(0));
    const auto tt = static_cast<int>(cost.size(1));
    const double* data = cost.data_ptr<double>();
    std::vector<double> flat(data,
                             data + static_cast<std::size_t>(q) * static_cast<std::size_t>(tt));

    auto pairs = core::LinearSumAssignment(flat, q, tt);  // (query, target), sorted by query
    std::vector<std::int64_t> src;
    std::vector<std::int64_t> dst;
    src.reserve(pairs.size());
    dst.reserve(pairs.size());
    for (const auto& [query, target] : pairs) {
      src.push_back(query);
      dst.push_back(target);
    }
    auto src_t = torch::tensor(src, idx_opts);
    auto dst_t = torch::tensor(dst, idx_opts);
    result.emplace_back(std::move(src_t), std::move(dst_t));
  }
  return result;
}

std::vector<MatchIndices> OneToManyMatch(const models::Detections& outputs,
                                         const TargetBatch& targets, int k,
                                         const MatchWeights& weights) {
  torch::NoGradGuard no_grad;
  const auto batch = outputs.logits.size(0);
  std::vector<MatchIndices> result;
  result.reserve(static_cast<std::size_t>(batch));
  const auto idx_opts = torch::TensorOptions().dtype(torch::kInt64);

  for (std::int64_t b = 0; b < batch; ++b) {
    const auto t = targets[static_cast<std::size_t>(b)];
    const auto num_targets = t.labels.size(0);
    if (num_targets == 0 || k <= 0) {
      result.emplace_back(torch::empty({0}, idx_opts), torch::empty({0}, idx_opts));
      continue;
    }
    auto cost = CostMatrix(outputs.logits[b], outputs.boxes[b], t.labels, t.boxes, weights);  // [Q,T]
    const auto kk = std::min<std::int64_t>(k, cost.size(0));
    // Each GT (column) keeps its kk lowest-cost queries (rows).
    auto top = std::get<1>((-cost).topk(kk, /*dim=*/0)).to(idx_opts);             // [kk, T] query idx
    auto src = top.reshape(-1);                                                    // [kk*T]
    auto dst = torch::arange(num_targets, idx_opts).unsqueeze(0).expand({kk, num_targets}).reshape(-1);
    result.emplace_back(src.contiguous(), dst.contiguous());
  }
  return result;
}

}  // namespace detr::train
