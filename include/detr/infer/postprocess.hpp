// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Turns DETR's raw outputs into detections. For each query we take the best
// non-"no object" class and its softmax score, convert the box from normalized
// cxcywh to absolute xywh against the original image size, and emit it. DETR is
// NMS-free, so all queries are kept. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <vector>

#include <torch/torch.h>

#include "detr/eval/coco_eval.hpp"
#include "detr/models/model.hpp"

namespace detr::infer {

std::vector<eval::DtBox> PostprocessImage(const models::Detections& outputs, int batch_index,
                                          int orig_w, int orig_h, int num_classes);

}  // namespace detr::infer
