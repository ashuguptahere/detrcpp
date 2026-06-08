// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/matcher.hpp"

#include <utility>
#include <vector>

#include "detr/core/assignment.hpp"
#include "detr/train/box_ops.hpp"

namespace detr::train {

std::vector<MatchIndices> HungarianMatch(const models::Detections& outputs,
                                         const TargetBatch& targets,
                                         const MatchWeights& weights) {
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

    auto logits = outputs.logits[b];                       // [Q, C+1]
    auto boxes = outputs.boxes[b];                         // [Q, 4] cxcywh
    auto labels = t.labels.to(logits.device());            // [T]
    auto tgt_boxes = t.boxes.to(boxes.device());           // [T, 4]

    auto prob = logits.softmax(-1);                        // [Q, C+1]
    auto cost_class = -prob.index_select(1, labels);       // [Q, T]
    auto cost_bbox = torch::cdist(boxes, tgt_boxes, /*p=*/1);  // [Q, T]
    auto cost_giou =
        -GeneralizedBoxIou(BoxCxcywhToXyxy(boxes), BoxCxcywhToXyxy(tgt_boxes));  // [Q, T]

    auto cost = weights.bbox * cost_bbox + weights.cls * cost_class + weights.giou * cost_giou;
    cost = cost.to(torch::kCPU, torch::kDouble).contiguous();  // [Q, T]

    const auto q = static_cast<int>(cost.size(0));
    const auto tt = static_cast<int>(cost.size(1));
    const double* data = cost.data_ptr<double>();
    std::vector<double> flat(data, data + static_cast<std::size_t>(q) * static_cast<std::size_t>(tt));

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

}  // namespace detr::train
