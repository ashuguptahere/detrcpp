// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/box_ops.hpp"

#include <utility>

namespace detr::train {

torch::Tensor BoxCxcywhToXyxy(const torch::Tensor& x) {
  auto cx = x.select(-1, 0);
  auto cy = x.select(-1, 1);
  auto w = x.select(-1, 2);
  auto h = x.select(-1, 3);
  return torch::stack({cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h}, -1);
}

torch::Tensor BoxXyxyToCxcywh(const torch::Tensor& x) {
  auto x0 = x.select(-1, 0);
  auto y0 = x.select(-1, 1);
  auto x1 = x.select(-1, 2);
  auto y1 = x.select(-1, 3);
  return torch::stack({(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)}, -1);
}

torch::Tensor BoxArea(const torch::Tensor& boxes) {
  return (boxes.select(-1, 2) - boxes.select(-1, 0)) * (boxes.select(-1, 3) - boxes.select(-1, 1));
}

std::pair<torch::Tensor, torch::Tensor> BoxIou(const torch::Tensor& a, const torch::Tensor& b) {
  auto area1 = BoxArea(a);                                         // [N]
  auto area2 = BoxArea(b);                                         // [M]
  auto a2 = a.unsqueeze(1);                                        // [N,1,4]
  auto b2 = b.unsqueeze(0);                                        // [1,M,4]
  auto lt = torch::max(a2.narrow(-1, 0, 2), b2.narrow(-1, 0, 2));  // [N,M,2]
  auto rb = torch::min(a2.narrow(-1, 2, 2), b2.narrow(-1, 2, 2));  // [N,M,2]
  auto wh = (rb - lt).clamp_min(0);                                // [N,M,2]
  auto inter = wh.select(-1, 0) * wh.select(-1, 1);                // [N,M]
  auto uni = area1.unsqueeze(1) + area2.unsqueeze(0) - inter;      // [N,M]
  auto iou = inter / uni.clamp_min(1e-7);
  return {iou, uni};
}

torch::Tensor GeneralizedBoxIou(const torch::Tensor& a, const torch::Tensor& b) {
  auto [iou, uni] = BoxIou(a, b);
  auto a2 = a.unsqueeze(1);
  auto b2 = b.unsqueeze(0);
  auto lt = torch::min(a2.narrow(-1, 0, 2), b2.narrow(-1, 0, 2));  // enclosing box
  auto rb = torch::max(a2.narrow(-1, 2, 2), b2.narrow(-1, 2, 2));
  auto wh = (rb - lt).clamp_min(0);
  auto area = wh.select(-1, 0) * wh.select(-1, 1);  // [N,M]
  return iou - (area - uni) / area.clamp_min(1e-7);
}

torch::Tensor GeneralizedBoxIouPaired(const torch::Tensor& a, const torch::Tensor& b) {
  // Same formula as GeneralizedBoxIou, but elementwise (no broadcast) over the
  // matched pairs a[i] vs b[i], so it returns [...] instead of an [M,M] matrix.
  auto area1 = BoxArea(a);                                       // [...]
  auto area2 = BoxArea(b);                                       // [...]
  auto lt = torch::max(a.narrow(-1, 0, 2), b.narrow(-1, 0, 2));  // [..., 2] intersection
  auto rb = torch::min(a.narrow(-1, 2, 2), b.narrow(-1, 2, 2));
  auto wh = (rb - lt).clamp_min(0);
  auto inter = wh.select(-1, 0) * wh.select(-1, 1);  // [...]
  auto uni = area1 + area2 - inter;
  auto iou = inter / uni.clamp_min(1e-7);
  auto lt_c = torch::min(a.narrow(-1, 0, 2), b.narrow(-1, 0, 2));  // enclosing box
  auto rb_c = torch::max(a.narrow(-1, 2, 2), b.narrow(-1, 2, 2));
  auto wh_c = (rb_c - lt_c).clamp_min(0);
  auto area_c = wh_c.select(-1, 0) * wh_c.select(-1, 1);  // [...]
  return iou - (area_c - uni) / area_c.clamp_min(1e-7);
}

}  // namespace detr::train
