// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Turns DETR's raw outputs into detections, converting boxes from normalized
// cxcywh to absolute xywh against the original image size. DETR is NMS-free.
// Softmax models (focal=false): best non-"no object" class per query, all kept.
// Focal models (focal=true): sigmoid scores, top-100 over all query x class pairs
// (a query may yield multiple detections). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <vector>

#include "detr/eval/coco_eval.hpp"
#include "detr/models/model.hpp"

namespace detr::infer {

std::vector<eval::DtBox> PostprocessImage(const models::Detections& outputs, int batch_index,
                                          int orig_w, int orig_h, int num_classes,
                                          bool focal = false);

}  // namespace detr::infer
