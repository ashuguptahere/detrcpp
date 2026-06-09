// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Runs a model over a dataset split and returns COCO metrics. Ground-truth boxes
// are taken in absolute pixels from each Sample's original (width, height), so
// the area-based small/medium/large breakdown is meaningful. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/data/dataset.hpp"
#include "detr/eval/coco_eval.hpp"
#include "detr/models/model.hpp"

namespace detr::eval {

// Evaluates up to |max_images| of the split (0 = all). When |aspect_preserve| is
// true, images are resized keeping aspect (shortest side = imgsz, longest <=
// |max_size|) and run one at a time — DETR's eval preprocessing — instead of the
// square-resize batched path.
CocoMetrics EvaluateModel(models::IModel& model, const data::Dataset& dataset,
                          data::Split split, int imgsz, int batch, torch::Device device,
                          int max_images = 0, bool aspect_preserve = false,
                          int max_size = 1333);

}  // namespace detr::eval
