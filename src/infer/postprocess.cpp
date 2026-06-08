// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/infer/postprocess.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

namespace detr::infer {

std::vector<eval::DtBox> PostprocessImage(const models::Detections& outputs, int batch_index,
                                          int orig_w, int orig_h, int num_classes) {
  torch::NoGradGuard no_grad;
  auto logits = outputs.logits[batch_index];                 // [Q, C+1]
  auto boxes = outputs.boxes[batch_index].to(torch::kCPU);   // [Q, 4] cxcywh
  auto prob = logits.softmax(-1).narrow(1, 0, num_classes);  // drop no-object
  auto best = prob.max(1);
  auto scores = std::get<0>(best).to(torch::kCPU);
  auto labels = std::get<1>(best).to(torch::kCPU);

  const auto q = boxes.size(0);
  auto ba = boxes.accessor<float, 2>();
  auto sa = scores.accessor<float, 1>();
  auto la = labels.accessor<std::int64_t, 1>();
  const float w_scale = static_cast<float>(orig_w > 0 ? orig_w : 1);
  const float h_scale = static_cast<float>(orig_h > 0 ? orig_h : 1);

  std::vector<eval::DtBox> dets;
  dets.reserve(static_cast<std::size_t>(q));
  for (std::int64_t i = 0; i < q; ++i) {
    const float cx = ba[i][0];
    const float cy = ba[i][1];
    const float w = ba[i][2];
    const float h = ba[i][3];
    eval::DtBox d;
    d.category_id = static_cast<int>(la[i]);
    d.x = (cx - w / 2.0F) * w_scale;
    d.y = (cy - h / 2.0F) * h_scale;
    d.w = w * w_scale;
    d.h = h * h_scale;
    d.score = sa[i];
    dets.push_back(d);
  }
  return dets;
}

}  // namespace detr::infer
