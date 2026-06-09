// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/criterion.hpp"

#include <vector>

#include "detr/train/box_ops.hpp"

namespace detr::train {

namespace F = torch::nn::functional;

Losses SetCriterion::Compute(const models::Detections& outputs, const TargetBatch& targets,
                             const std::vector<MatchIndices>& matches) const {
  const auto device = outputs.logits.device();
  const auto batch = outputs.logits.size(0);
  const auto queries = outputs.logits.size(1);

  // (1) Classification: target class per (image, query); default = no-object.
  auto target_classes =
      torch::full({batch, queries}, static_cast<std::int64_t>(num_classes_),
                  torch::TensorOptions().dtype(torch::kInt64).device(device));
  for (std::int64_t b = 0; b < batch; ++b) {
    const auto& [src_idx, tgt_idx] = matches[static_cast<std::size_t>(b)];
    if (src_idx.numel() == 0) {
      continue;
    }
    auto labels_b =
        targets[static_cast<std::size_t>(b)].labels.to(device).index_select(0, tgt_idx);
    target_classes[b].index_copy_(0, src_idx, labels_b);
  }

  // Count target boxes (also the focal normalizer).
  std::int64_t num_boxes_total = 0;
  for (std::int64_t b = 0; b < batch; ++b) {
    num_boxes_total += matches[static_cast<std::size_t>(b)].first.numel();
  }

  torch::Tensor loss_ce;
  if (focal_) {
    // Sigmoid focal loss over num_classes (one-hot targets; background -> all 0).
    auto onehot = torch::zeros({batch, queries, num_classes_ + 1}, outputs.logits.options());
    onehot.scatter_(2, target_classes.unsqueeze(-1), 1.0);
    onehot = onehot.narrow(2, 0, num_classes_);  // drop the background column
    const double nb = static_cast<double>(num_boxes_total > 0 ? num_boxes_total : 1);
    auto prob = outputs.logits.sigmoid();
    auto ce = F::binary_cross_entropy_with_logits(
        outputs.logits, onehot, F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
    auto p_t = prob * onehot + (1 - prob) * (1 - onehot);
    auto loss = ce * (1 - p_t).pow(w_.focal_gamma);
    auto alpha_t = w_.focal_alpha * onehot + (1 - w_.focal_alpha) * (1 - onehot);
    loss = alpha_t * loss;
    loss_ce = loss.mean(1).sum() / nb * static_cast<double>(queries);
  } else {
    auto weight = torch::ones({num_classes_ + 1}, outputs.logits.options());
    weight[num_classes_] = w_.eos_coef;
    // cross_entropy wants [N, C, ...] vs [N, ...]; use [B, C+1, Q] vs [B, Q].
    loss_ce = F::cross_entropy(outputs.logits.transpose(1, 2), target_classes,
                               F::CrossEntropyFuncOptions().weight(weight));
  }

  // (2,3) Box losses over matched pairs, normalized by number of target boxes.
  std::vector<torch::Tensor> src_list;
  std::vector<torch::Tensor> tgt_list;
  std::int64_t num_boxes = 0;
  for (std::int64_t b = 0; b < batch; ++b) {
    const auto& [src_idx, tgt_idx] = matches[static_cast<std::size_t>(b)];
    if (src_idx.numel() == 0) {
      continue;
    }
    src_list.push_back(outputs.boxes[b].index_select(0, src_idx));
    tgt_list.push_back(
        targets[static_cast<std::size_t>(b)].boxes.to(device).index_select(0, tgt_idx));
    num_boxes += src_idx.numel();
  }
  const double norm = static_cast<double>(num_boxes > 0 ? num_boxes : 1);

  Losses out;
  out.loss_ce = loss_ce;
  if (src_list.empty()) {
    out.loss_bbox = torch::zeros({}, outputs.logits.options());
    out.loss_giou = torch::zeros({}, outputs.logits.options());
  } else {
    auto src_boxes = torch::cat(src_list, 0);  // [M,4] cxcywh
    auto tgt_boxes = torch::cat(tgt_list, 0);  // [M,4]
    out.loss_bbox =
        F::l1_loss(src_boxes, tgt_boxes, F::L1LossFuncOptions().reduction(torch::kNone))
            .sum() / norm;
    auto giou = GeneralizedBoxIou(BoxCxcywhToXyxy(src_boxes), BoxCxcywhToXyxy(tgt_boxes));
    out.loss_giou = (1.0 - giou.diagonal()).sum() / norm;
  }

  out.total = w_.cls * out.loss_ce + w_.bbox * out.loss_bbox + w_.giou * out.loss_giou;
  return out;
}

}  // namespace detr::train
